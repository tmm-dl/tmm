/**
 * @file trainer.cpp
 * @brief Trainer implementation — full training loop with lifecycle events.
 */

#include <tmm/compat/format.hpp>
#include <tmm/plugins/abi.h>
#include <tmm/trainer/trainer.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <string>

namespace tmm::trainer {

	namespace {

		/// Serialize a float to a JSON value; NaN / Inf → "null".
		std::string fj(float v) {
			if (std::isnan(v) || std::isinf(v))
				return "null";
			return std::format("{:.6g}", v);
		}

	} // namespace

	/* =========================================================================
	 * Constructor & fluent setters
	 * ====================================================================== */

	Trainer::Trainer(
			conf::TrainingConfig config, plugins::PluginManager& plugins, std::unique_ptr<IModel> model,
			DatasetFactory train
	)
			: config_(std::move(config)), plugins_(plugins), model_(std::move(model)), trainFactory_(std::move(train)) {
	}

	Trainer& Trainer::optimizer(std::unique_ptr<IOptimizer> opt) {
		optimizer_ = std::move(opt);
		return *this;
	}

	Trainer& Trainer::scheduler(std::unique_ptr<ILRScheduler> sched) {
		scheduler_ = std::move(sched);
		return *this;
	}

	Trainer& Trainer::validation(DatasetFactory val) {
		valFactory_ = std::move(val);
		return *this;
	}

	Trainer& Trainer::addCallback(std::unique_ptr<Callback> cb) {
		callbacks_.push_back(std::move(cb));
		return *this;
	}

	Trainer& Trainer::stopPredicate(std::function<bool()> pred) {
		stopPredicate_ = std::move(pred);
		return *this;
	}

	/* =========================================================================
	 * JSON helpers
	 * ====================================================================== */

	std::string Trainer::buildFitBeginJson() const {
#ifndef TMM_VERSION
#define TMM_VERSION "unknown"
#endif
		return std::format(
				R"({{"tmm_version":"{}","total_epochs":{},"dataset_uri":"{}","dataset_split":"{}","model_path":"{}"}})",
				TMM_VERSION, config_.epochs, config_.dataset.uri, config_.dataset.split, config_.model.path
		);
	}

	std::string Trainer::buildBatchJson(int64_t epoch, int64_t step, float loss, float lr) const {
		return std::format(R"({{"epoch":{},"step":{},"loss":{},"learning_rate":{}}})", epoch, step, fj(loss), fj(lr));
	}

	std::string Trainer::buildEpochJson(const EpochMetrics& m) const {
		return std::format(
				R"({{"epoch":{},"step":{},"train_loss":{},"val_loss":{},"learning_rate":{},"throughput":{:.1f}}})",
				m.epoch, m.globalStep, fj(m.trainLoss), fj(m.valLoss), fj(m.learningRate), m.throughputSamplesPerSec
		);
	}

	/* =========================================================================
	 * Validation pass
	 * ====================================================================== */

	std::expected<float, std::string> Trainer::runValidation() {
		auto iterResult = valFactory_();
		if (!iterResult)
			return std::unexpected(iterResult.error());

		double totalLoss = 0.0;
		int64_t batches = 0;
		std::shared_ptr<arrow::RecordBatch> raw;

		while ((*iterResult)->next(raw)) {
			const Batch b{raw, batches, batches};
			totalLoss += model_->infer(b).loss;
			++batches;
		}

		return batches > 0 ? static_cast<float>(totalLoss / static_cast<double>(batches)) : 0.0f;
	}

	/* =========================================================================
	 * Main training loop
	 * ====================================================================== */

	void Trainer::log(std::string_view key, float value) {
		currentMetrics_[std::string(key)] = value;
		plugins_.emitMetric(key, value, static_cast<int32_t>(globalStep_));
		const CallbackMetrics cm(currentMetrics_);
		for (auto& cb : callbacks_) {
			cb->on_log(*this, key, value, static_cast<int32_t>(globalStep_));
		}
	}

	std::expected<EpochMetrics, std::string> Trainer::fit() {
		// ── One-time: load config plugins + instantiate config callbacks ──────
		if (!configApplied_) {
			configApplied_ = true;

			for (const auto& pe : config_.plugins) {
				if (pe.path.empty())
					continue; // already loaded by caller via name resolution
				if (auto r = plugins_.load(pe.path, pe.config); !r) {
					plugins_.emitLog(TMM_LOG_WARN, std::format("plugin load failed ({}): {}", pe.path, r.error()));
				}
			}

			for (const auto& ce : config_.callbacks) {
				std::string err;
				auto cb = plugins_.makeCallback(ce.type, ce.config, &err);
				if (cb) {
					callbacks_.push_back(std::move(cb));
				} else {
					plugins_.emitLog(
							TMM_LOG_WARN, std::format("callback '{}' not found or failed to create: {}", ce.type, err)
					);
				}
			}
		}

		plugins_.emitFitBegin(buildFitBeginJson());
		{
			const CallbackMetrics cm(currentMetrics_);
			for (auto& cb : callbacks_)
				cb->on_fit_begin(*this, cm);
		}

		EpochMetrics finalMetrics;
		globalStep_ = 0;		   ///< Batch-level counter — also exposed via log()
		int64_t globalOptStep = 0; ///< Optimizer-step counter
		float currentLr = optimizer_ ? optimizer_->learningRate() : config_.optimizer.lr;

		for (int64_t epoch = 1; epoch <= config_.epochs; ++epoch) {
			plugins_.emitEpochBegin(static_cast<uint32_t>(epoch), static_cast<uint32_t>(config_.epochs));
			for (auto& cb : callbacks_)
				cb->on_epoch_begin(*this, epoch, config_.epochs);

			// ── Fresh iterator for this epoch ──────────────────────────────
			auto iterResult = trainFactory_();
			if (!iterResult)
				return std::unexpected(iterResult.error());
			auto& iter = *iterResult;

			double epochLoss = 0.0;
			int64_t epochSamples = 0;
			int64_t accumStep = 0;
			int64_t batchIdx = 0;

			model_->zeroGrad();
			const auto epochStart = std::chrono::steady_clock::now();

			// ── Batch loop ─────────────────────────────────────────────────
			std::shared_ptr<arrow::RecordBatch> raw;
			while (iter->next(raw)) {
				// Check external stop request (e.g. SIGINT from main)
				if (stopPredicate_ && stopPredicate_()) {
					plugins_.emitLog(TMM_LOG_WARN, "training interrupted by user");
					goto fit_interrupted;
				}

				const Batch batch{raw, globalStep_, batchIdx};

				plugins_.emitBatchBegin(
						static_cast<uint32_t>(batchIdx), 0 /* total unknown without pre-scan */
				);

				const auto stepOut = model_->step(batch);
				if (stepOut.interrupted) {
					plugins_.emitLog(TMM_LOG_WARN, "training interrupted by user");
					goto fit_interrupted;
				}
				const float rawLoss = stepOut.loss;
				const float loss = plugins_.emitLossComputed(rawLoss);

				epochLoss += loss;
				epochSamples += raw->num_rows();
				++accumStep;
				++globalStep_;
				++batchIdx;

				// ── Optimizer step ─────────────────────────────────────────
				if (accumStep == config_.gradientAccumulationSteps) {
					if (optimizer_)
						optimizer_->step();
					++globalOptStep;

					if (scheduler_) {
						currentLr = scheduler_->step(globalOptStep - 1);
						if (optimizer_)
							optimizer_->setLearningRate(currentLr);
					}

					model_->zeroGrad();
					accumStep = 0;
				}

				// ── Per-batch metric logging ───────────────────────────────
				currentMetrics_["train_loss"] = loss;
				currentMetrics_["learning_rate"] = currentLr;
				plugins_.emitMetric("train_loss", loss, static_cast<int32_t>(globalStep_));
				plugins_.emitMetric("learning_rate", currentLr, static_cast<int32_t>(globalStep_));

				plugins_.emitBatchEnd(
						static_cast<uint32_t>(batchIdx - 1), loss, buildBatchJson(epoch, globalStep_, loss, currentLr)
				);
			}

			// ── Flush partial accumulation window ──────────────────────────
			if (accumStep > 0 && optimizer_) {
				optimizer_->step();
				++globalOptStep;
				if (scheduler_) {
					currentLr = scheduler_->step(globalOptStep - 1);
					if (optimizer_)
						optimizer_->setLearningRate(currentLr);
				}
				model_->zeroGrad();
			}

			const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - epochStart).count();

			const float avgLoss = batchIdx > 0 ? static_cast<float>(epochLoss / static_cast<double>(batchIdx)) : 0.0f;
			const double throughput = elapsed > 0.0 ? static_cast<double>(epochSamples) / elapsed : 0.0;

			// ── Validation ────────────────────────────────────────────────
			float valLoss = std::numeric_limits<float>::quiet_NaN();
			if (valFactory_) {
				if (auto vr = runValidation(); vr) {
					valLoss = *vr;
					currentMetrics_["val_loss"] = valLoss;
					plugins_.emitMetric("val_loss", valLoss, static_cast<int32_t>(globalStep_));
				} else {
					plugins_.emitLog(TMM_LOG_WARN, "validation failed: " + vr.error());
				}
				plugins_.emitValidationEnd(
						buildEpochJson({epoch, globalStep_, avgLoss, valLoss, currentLr, throughput})
				);
				{
					const CallbackMetrics cm(currentMetrics_);
					for (auto& cb : callbacks_)
						cb->on_validation_end(*this, cm);
				}
			}

			EpochMetrics metrics{epoch, globalStep_, avgLoss, valLoss, currentLr, throughput};
			finalMetrics = metrics;

			// ── Per-epoch metric logging ───────────────────────────────────
			currentMetrics_["epoch"] = static_cast<float>(epoch);
			currentMetrics_["epoch_train_loss"] = avgLoss;
			currentMetrics_["throughput"] = static_cast<float>(throughput);
			plugins_.emitMetric("epoch_train_loss", avgLoss, static_cast<int32_t>(globalStep_));
			plugins_.emitMetric("throughput", static_cast<float>(throughput), static_cast<int32_t>(globalStep_));

			// ── Log ───────────────────────────────────────────────────────
			if (std::isnan(valLoss)) {
				plugins_.emitLog(
						TMM_LOG_INFO, std::format(
											  "epoch {}/{} — loss: {:.4f}  lr: {:.2e}  {:.0f} samples/s", epoch,
											  config_.epochs, avgLoss, currentLr, throughput
									  )
				);
			} else {
				plugins_.emitLog(
						TMM_LOG_INFO,
						std::format(
								"epoch {}/{} — loss: {:.4f}  val_loss: {:.4f}  lr: {:.2e}  {:.0f} samples/s", epoch,
								config_.epochs, avgLoss, valLoss, currentLr, throughput
						)
				);
			}

			bool stop = plugins_.emitEpochEnd(static_cast<uint32_t>(epoch), buildEpochJson(metrics));
			{
				const CallbackMetrics cm(currentMetrics_);
				for (auto& cb : callbacks_) {
					if (cb->on_epoch_end(*this, epoch, cm))
						stop = true;
				}
			}
			if (stop) {
				plugins_.emitLog(TMM_LOG_INFO, "early stopping triggered");
				break;
			}
		}
	fit_interrupted:;

		plugins_.emitFitEnd(buildEpochJson(finalMetrics));
		{
			const CallbackMetrics cm(currentMetrics_);
			for (auto& cb : callbacks_)
				cb->on_fit_end(*this, cm);
		}
		return finalMetrics;
	}

} // namespace tmm::trainer

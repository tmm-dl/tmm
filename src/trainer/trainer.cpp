/**
 * @file trainer.cpp
 * @brief Trainer implementation — full training loop with lifecycle events.
 */

#include <ttm/compat/format.hpp>
#include <ttm/plugins/abi.h>
#include <ttm/trainer/trainer.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <string>

namespace ttm::trainer {

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
			: config_(std::move(config)), plugins_(plugins), model_(std::move(model)),
			  train_factory_(std::move(train)) {}

	Trainer& Trainer::optimizer(std::unique_ptr<IOptimizer> opt) {
		optimizer_ = std::move(opt);
		return *this;
	}

	Trainer& Trainer::scheduler(std::unique_ptr<ILRScheduler> sched) {
		scheduler_ = std::move(sched);
		return *this;
	}

	Trainer& Trainer::validation(DatasetFactory val) {
		val_factory_ = std::move(val);
		return *this;
	}

	Trainer& Trainer::add_callback(std::unique_ptr<Callback> cb) {
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

	std::string Trainer::build_fit_begin_json() const {
#ifndef TTM_VERSION
#define TTM_VERSION "unknown"
#endif
		return std::format(
				R"({{"ttm_version":"{}","total_epochs":{},"dataset_uri":"{}","dataset_split":"{}","model_path":"{}"}})",
				TTM_VERSION, config_.epochs, config_.dataset.uri, config_.dataset.split, config_.model.path
		);
	}

	std::string Trainer::build_batch_json(int64_t epoch, int64_t step, float loss, float lr) const {
		return std::format(R"({{"epoch":{},"step":{},"loss":{},"learning_rate":{}}})", epoch, step, fj(loss), fj(lr));
	}

	std::string Trainer::build_epoch_json(const EpochMetrics& m) const {
		return std::format(
				R"({{"epoch":{},"step":{},"train_loss":{},"val_loss":{},"learning_rate":{},"throughput":{:.1f}}})",
				m.epoch, m.global_step, fj(m.train_loss), fj(m.val_loss), fj(m.learning_rate),
				m.throughput_samples_per_sec
		);
	}

	/* =========================================================================
	 * Validation pass
	 * ====================================================================== */

	std::expected<float, std::string> Trainer::run_validation() {
		auto iterResult = val_factory_();
		if (!iterResult)
			return std::unexpected(iterResult.error());

		double total_loss = 0.0;
		int64_t batches = 0;
		std::shared_ptr<arrow::RecordBatch> raw;

		while ((*iterResult)->next(raw)) {
			const Batch b{raw, batches, batches};
			total_loss += model_->infer(b).loss;
			++batches;
		}

		return batches > 0 ? static_cast<float>(total_loss / static_cast<double>(batches)) : 0.0f;
	}

	/* =========================================================================
	 * Main training loop
	 * ====================================================================== */

	void Trainer::log(std::string_view key, float value) {
		currentMetrics_[std::string(key)] = value;
		plugins_.emit_metric(key, value, static_cast<int32_t>(globalStep_));
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
					plugins_.emit_log(TTM_LOG_WARN, std::format("plugin load failed ({}): {}", pe.path, r.error()));
				}
			}

			for (const auto& ce : config_.callbacks) {
				std::string err;
				auto cb = plugins_.make_callback(ce.type, ce.config, &err);
				if (cb) {
					callbacks_.push_back(std::move(cb));
				} else {
					plugins_.emit_log(
							TTM_LOG_WARN, std::format("callback '{}' not found or failed to create: {}", ce.type, err)
					);
				}
			}
		}

		plugins_.emit_fit_begin(build_fit_begin_json());
		{
			const CallbackMetrics cm(currentMetrics_);
			for (auto& cb : callbacks_)
				cb->on_fit_begin(*this, cm);
		}

		EpochMetrics final_metrics;
		globalStep_ = 0;			 ///< Batch-level counter — also exposed via log()
		int64_t global_opt_step = 0; ///< Optimizer-step counter
		float current_lr = optimizer_ ? optimizer_->learning_rate() : config_.optimizer.lr;

		for (int64_t epoch = 1; epoch <= config_.epochs; ++epoch) {
			plugins_.emit_epoch_begin(static_cast<uint32_t>(epoch), static_cast<uint32_t>(config_.epochs));
			for (auto& cb : callbacks_)
				cb->on_epoch_begin(*this, epoch, config_.epochs);

			// ── Fresh iterator for this epoch ──────────────────────────────
			auto iterResult = train_factory_();
			if (!iterResult)
				return std::unexpected(iterResult.error());
			auto& iter = *iterResult;

			double epoch_loss = 0.0;
			int64_t epoch_samples = 0;
			int64_t accum_step = 0;
			int64_t batch_idx = 0;

			model_->zero_grad();
			const auto epoch_start = std::chrono::steady_clock::now();

			// ── Batch loop ─────────────────────────────────────────────────
			std::shared_ptr<arrow::RecordBatch> raw;
			while (iter->next(raw)) {
				// Check external stop request (e.g. SIGINT from main)
				if (stopPredicate_ && stopPredicate_()) {
					plugins_.emit_log(TTM_LOG_WARN, "training interrupted by user");
					goto fit_interrupted;
				}

				const Batch batch{raw, globalStep_, batch_idx};

				plugins_.emit_batch_begin(
						static_cast<uint32_t>(batch_idx), 0 /* total unknown without pre-scan */
				);

				const auto step_out = model_->step(batch);
				if (step_out.interrupted) {
					plugins_.emit_log(TTM_LOG_WARN, "training interrupted by user");
					goto fit_interrupted;
				}
				const float raw_loss = step_out.loss;
				const float loss = plugins_.emit_loss_computed(raw_loss);

				epoch_loss += loss;
				epoch_samples += raw->num_rows();
				++accum_step;
				++globalStep_;
				++batch_idx;

				// ── Optimizer step ─────────────────────────────────────────
				if (accum_step == config_.gradient_accumulation_steps) {
					if (optimizer_)
						optimizer_->step();
					++global_opt_step;

					if (scheduler_) {
						current_lr = scheduler_->step(global_opt_step - 1);
						if (optimizer_)
							optimizer_->set_learning_rate(current_lr);
					}

					model_->zero_grad();
					accum_step = 0;
				}

				// ── Per-batch metric logging ───────────────────────────────
				currentMetrics_["train_loss"] = loss;
				currentMetrics_["learning_rate"] = current_lr;
				plugins_.emit_metric("train_loss", loss, static_cast<int32_t>(globalStep_));
				plugins_.emit_metric("learning_rate", current_lr, static_cast<int32_t>(globalStep_));

				plugins_.emit_batch_end(
						static_cast<uint32_t>(batch_idx - 1), loss,
						build_batch_json(epoch, globalStep_, loss, current_lr)
				);
			}

			// ── Flush partial accumulation window ──────────────────────────
			if (accum_step > 0 && optimizer_) {
				optimizer_->step();
				++global_opt_step;
				if (scheduler_) {
					current_lr = scheduler_->step(global_opt_step - 1);
					if (optimizer_)
						optimizer_->set_learning_rate(current_lr);
				}
				model_->zero_grad();
			}

			const double elapsed =
					std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();

			const float avg_loss =
					batch_idx > 0 ? static_cast<float>(epoch_loss / static_cast<double>(batch_idx)) : 0.0f;
			const double throughput = elapsed > 0.0 ? static_cast<double>(epoch_samples) / elapsed : 0.0;

			// ── Validation ────────────────────────────────────────────────
			float val_loss = std::numeric_limits<float>::quiet_NaN();
			if (val_factory_) {
				if (auto vr = run_validation(); vr) {
					val_loss = *vr;
					currentMetrics_["val_loss"] = val_loss;
					plugins_.emit_metric("val_loss", val_loss, static_cast<int32_t>(globalStep_));
				} else {
					plugins_.emit_log(TTM_LOG_WARN, "validation failed: " + vr.error());
				}
				plugins_.emit_validation_end(
						build_epoch_json({epoch, globalStep_, avg_loss, val_loss, current_lr, throughput})
				);
				{
					const CallbackMetrics cm(currentMetrics_);
					for (auto& cb : callbacks_)
						cb->on_validation_end(*this, cm);
				}
			}

			EpochMetrics metrics{epoch, globalStep_, avg_loss, val_loss, current_lr, throughput};
			final_metrics = metrics;

			// ── Per-epoch metric logging ───────────────────────────────────
			currentMetrics_["epoch"] = static_cast<float>(epoch);
			currentMetrics_["epoch_train_loss"] = avg_loss;
			currentMetrics_["throughput"] = static_cast<float>(throughput);
			plugins_.emit_metric("epoch_train_loss", avg_loss, static_cast<int32_t>(globalStep_));
			plugins_.emit_metric("throughput", static_cast<float>(throughput), static_cast<int32_t>(globalStep_));

			// ── Log ───────────────────────────────────────────────────────
			if (std::isnan(val_loss)) {
				plugins_.emit_log(
						TTM_LOG_INFO, std::format(
											  "epoch {}/{} — loss: {:.4f}  lr: {:.2e}  {:.0f} samples/s", epoch,
											  config_.epochs, avg_loss, current_lr, throughput
									  )
				);
			} else {
				plugins_.emit_log(
						TTM_LOG_INFO,
						std::format(
								"epoch {}/{} — loss: {:.4f}  val_loss: {:.4f}  lr: {:.2e}  {:.0f} samples/s", epoch,
								config_.epochs, avg_loss, val_loss, current_lr, throughput
						)
				);
			}

			bool stop = plugins_.emit_epoch_end(static_cast<uint32_t>(epoch), build_epoch_json(metrics));
			{
				const CallbackMetrics cm(currentMetrics_);
				for (auto& cb : callbacks_) {
					if (cb->on_epoch_end(*this, epoch, cm))
						stop = true;
				}
			}
			if (stop) {
				plugins_.emit_log(TTM_LOG_INFO, "early stopping triggered");
				break;
			}
		}
	fit_interrupted:;

		plugins_.emit_fit_end(build_epoch_json(final_metrics));
		{
			const CallbackMetrics cm(currentMetrics_);
			for (auto& cb : callbacks_)
				cb->on_fit_end(*this, cm);
		}
		return final_metrics;
	}

} // namespace ttm::trainer

/**
 * @file trainer.cpp
 * @brief Trainer implementation — full training loop with lifecycle events.
 */

#include <ttm/trainer/trainer.hpp>
#include <ttm/compat/format.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

namespace ttm::trainer {

	namespace {

		/// Serialize a float to a JSON value; NaN / Inf → "null".
		std::string fj(float v) {
			if (std::isnan(v) || std::isinf(v)) return "null";
			return std::format("{:.6g}", v);
		}

	} // namespace

	/* =========================================================================
	 * Constructor & fluent setters
	 * ====================================================================== */

	Trainer::Trainer(
		conf::TrainingConfig    config,
		plugins::PluginManager& plugins,
		std::unique_ptr<IModel> model,
		DatasetFactory          train
	)
		: config_(std::move(config))
		, plugins_(plugins)
		, model_(std::move(model))
		, train_factory_(std::move(train))
	{}

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

	/* =========================================================================
	 * JSON helpers
	 * ====================================================================== */

	std::string Trainer::build_fit_begin_json() const {
#ifndef TTM_VERSION
#	define TTM_VERSION "unknown"
#endif
		return std::format(
			R"({{"ttm_version":"{}","total_epochs":{},"dataset_uri":"{}","dataset_split":"{}"}})",
			TTM_VERSION,
			config_.epochs,
			config_.dataset.uri,
			config_.dataset.split
		);
	}

	std::string Trainer::build_batch_json(
		int64_t epoch, int64_t step, float loss, float lr
	) const {
		return std::format(
			R"({{"epoch":{},"step":{},"loss":{},"learning_rate":{}}})",
			epoch, step, fj(loss), fj(lr)
		);
	}

	std::string Trainer::build_epoch_json(const EpochMetrics& m) const {
		return std::format(
			R"({{"epoch":{},"step":{},"train_loss":{},"val_loss":{},"learning_rate":{},"throughput":{:.1f}}})",
			m.epoch, m.global_step,
			fj(m.train_loss), fj(m.val_loss), fj(m.learning_rate),
			m.throughput_samples_per_sec
		);
	}

	/* =========================================================================
	 * Validation pass
	 * ====================================================================== */

	std::expected<float, std::string> Trainer::run_validation() {
		auto iterResult = val_factory_();
		if (!iterResult) return std::unexpected(iterResult.error());

		double  total_loss = 0.0;
		int64_t batches    = 0;
		std::shared_ptr<arrow::RecordBatch> raw;

		while ((*iterResult)->next(raw)) {
			const Batch b{raw, batches, batches};
			total_loss += model_->infer(b).loss;
			++batches;
		}

		return batches > 0
			? static_cast<float>(total_loss / static_cast<double>(batches))
			: 0.0f;
	}

	/* =========================================================================
	 * Main training loop
	 * ====================================================================== */

	std::expected<EpochMetrics, std::string> Trainer::fit() {
		plugins_.emit_fit_begin(build_fit_begin_json());

		EpochMetrics final_metrics;
		int64_t      global_step     = 0; ///< Batch-level counter (incremented every batch)
		int64_t      global_opt_step = 0; ///< Optimizer-step counter
		float        current_lr      = optimizer_
			? optimizer_->learning_rate()
			: config_.optimizer.lr;

		for (int64_t epoch = 1; epoch <= config_.epochs; ++epoch) {
			plugins_.emit_epoch_begin(
				static_cast<uint32_t>(epoch),
				static_cast<uint32_t>(config_.epochs)
			);

			// ── Fresh iterator for this epoch ──────────────────────────────
			auto iterResult = train_factory_();
			if (!iterResult) return std::unexpected(iterResult.error());
			auto& iter = *iterResult;

			double  epoch_loss    = 0.0;
			int64_t epoch_samples = 0;
			int64_t accum_step    = 0;
			int64_t batch_idx     = 0;

			model_->zero_grad();
			const auto epoch_start = std::chrono::steady_clock::now();

			// ── Batch loop ─────────────────────────────────────────────────
			std::shared_ptr<arrow::RecordBatch> raw;
			while (iter->next(raw)) {
				const Batch batch{raw, global_step, batch_idx};

				plugins_.emit_batch_begin(
					static_cast<uint32_t>(batch_idx),
					0 /* total unknown without pre-scan */
				);

				const float raw_loss = model_->step(batch).loss;
				const float loss     = plugins_.emit_loss_computed(raw_loss);

				epoch_loss    += loss;
				epoch_samples += raw->num_rows();
				++accum_step;
				++global_step;
				++batch_idx;

				// ── Optimizer step ─────────────────────────────────────────
				if (accum_step == config_.gradient_accumulation_steps) {
					if (optimizer_) optimizer_->step();
					++global_opt_step;

					if (scheduler_) {
						current_lr = scheduler_->step(global_opt_step - 1);
						if (optimizer_) optimizer_->set_learning_rate(current_lr);
					}

					model_->zero_grad();
					accum_step = 0;
				}

				plugins_.emit_batch_end(
					static_cast<uint32_t>(batch_idx - 1),
					loss,
					build_batch_json(epoch, global_step, loss, current_lr)
				);
			}

			// ── Flush partial accumulation window ──────────────────────────
			if (accum_step > 0 && optimizer_) {
				optimizer_->step();
				++global_opt_step;
				if (scheduler_) {
					current_lr = scheduler_->step(global_opt_step - 1);
					if (optimizer_) optimizer_->set_learning_rate(current_lr);
				}
				model_->zero_grad();
			}

			const double elapsed = std::chrono::duration<double>(
				std::chrono::steady_clock::now() - epoch_start
			).count();

			const float  avg_loss  = batch_idx > 0
				? static_cast<float>(epoch_loss / static_cast<double>(batch_idx))
				: 0.0f;
			const double throughput = elapsed > 0.0
				? static_cast<double>(epoch_samples) / elapsed
				: 0.0;

			// ── Validation ────────────────────────────────────────────────
			float val_loss = std::numeric_limits<float>::quiet_NaN();
			if (val_factory_) {
				if (auto vr = run_validation(); vr) {
					val_loss = *vr;
				} else {
					std::cerr << "[ttm] validation failed: " << vr.error() << '\n';
				}
				plugins_.emit_validation_end(
					build_epoch_json({epoch, global_step, avg_loss, val_loss, current_lr, throughput})
				);
			}

			EpochMetrics metrics{epoch, global_step, avg_loss, val_loss, current_lr, throughput};
			final_metrics = metrics;

			// ── Log ───────────────────────────────────────────────────────
			if (std::isnan(val_loss)) {
				std::cout << std::format(
					"[ttm] epoch {}/{} — loss: {:.4f}  lr: {:.2e}  {:.0f} samples/s\n",
					epoch, config_.epochs, avg_loss, current_lr, throughput
				);
			} else {
				std::cout << std::format(
					"[ttm] epoch {}/{} — loss: {:.4f}  val_loss: {:.4f}  lr: {:.2e}  {:.0f} samples/s\n",
					epoch, config_.epochs, avg_loss, val_loss, current_lr, throughput
				);
			}

			if (plugins_.emit_epoch_end(
					static_cast<uint32_t>(epoch),
					build_epoch_json(metrics)
				)) {
				std::cout << "[ttm] early stopping requested by plugin\n";
				break;
			}
		}

		plugins_.emit_fit_end(build_epoch_json(final_metrics));
		return final_metrics;
	}

} // namespace ttm::trainer

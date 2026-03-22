/**
 * @file trainer.hpp
 * @brief Trainer — orchestrates the full training loop.
 *
 * @details
 * The Trainer owns the model, optimizer, and optional LR scheduler.  It
 * coordinates with the @ref tmm::plugins::PluginManager for lifecycle events
 * and iterates the dataset epoch by epoch via a @ref DatasetFactory.
 *
 * ### Usage
 * @code{.cpp}
 * // Build the trainer, chain optional components, then run.
 * auto result = tmm::trainer::Trainer(cfg, plugins, std::move(model), trainFactory)
 *     .optimizer(std::make_unique<MyAdam>(cfg.optimizer.lr))
 *     .scheduler(std::make_unique<MyCosineWarmup>(warmupSteps, totalSteps))
 *     .validation(valFactory)
 *     .fit();
 *
 * if (!result) {
 *     std::cerr << "Training failed: " << result.error() << '\n';
 * }
 * @endcode
 *
 * ### DatasetFactory
 * A `DatasetFactory` is a callable that returns a fresh @ref DatasetIterator.
 * It is invoked once at the start of each epoch so the Trainer does not need
 * to know how the dataset is stored or whether it can be rewound.
 */

#pragma once

#include <tmm/compat/expected.hpp>
#include <tmm/conf/config.hpp>
#include <tmm/datasets/dataset_loader.hpp>
#include <tmm/plugins/plugin_manager.hpp>
#include <tmm/trainer/callback.hpp>
#include <tmm/trainer/interfaces.hpp>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tmm::trainer {

	/**
	 * @brief Factory function that creates a fresh @ref DatasetIterator.
	 *
	 * Called once per training epoch (and once per validation pass).  The
	 * returned iterator is consumed fully and then discarded.
	 */
	using DatasetFactory = std::function<std::expected<std::unique_ptr<tmm::datasets::DatasetIterator>, std::string>()>;

	/**
	 * @brief Orchestrates the training loop over multiple epochs.
	 *
	 * @details
	 * Responsibilities:
	 * - Advance the DatasetIterator epoch by epoch.
	 * - Call @c IModel::step() for each batch and accumulate gradients.
	 * - Call @c IOptimizer::step() and @c ILRScheduler::step() at the right cadence.
	 * - Emit lifecycle events to the PluginManager at every major boundary.
	 * - Run a validation pass at the end of each epoch (if a val factory is set).
	 * - Log per-epoch progress to stdout.
	 */
	class Trainer {
	public:
		/**
		 * @param config       Resolved training configuration.
		 * @param plugins      Plugin manager (lifetime must exceed Trainer).
		 * @param model        Owning pointer to the model implementation.
		 * @param train        Factory that produces a fresh training-split iterator.
		 */
		Trainer(conf::TrainingConfig config, plugins::PluginManager& plugins, std::unique_ptr<IModel> model,
				DatasetFactory train);

		Trainer(Trainer&&) = default;
		Trainer& operator=(Trainer&&) = default;
		Trainer(const Trainer&) = delete;
		Trainer& operator=(const Trainer&) = delete;
		~Trainer() = default;

		/// Attach an optimizer.  Without one the model is expected to update itself.
		Trainer& optimizer(std::unique_ptr<IOptimizer> opt);

		/// Attach an LR scheduler.  Without one the optimizer's initial LR is kept constant.
		Trainer& scheduler(std::unique_ptr<ILRScheduler> sched);

		/// Attach a validation dataset factory (optional).
		Trainer& validation(DatasetFactory val);

		/// Attach a callback.  Callbacks are invoked in insertion order.
		Trainer& addCallback(std::unique_ptr<Callback> cb);

		/// Set a predicate polled between batches; returning true triggers graceful stop.
		Trainer& stopPredicate(std::function<bool()> pred);

		/**
		 * @brief Run the training loop.
		 * @return Final epoch metrics on success, or an error string on failure.
		 */
		[[nodiscard]] std::expected<EpochMetrics, std::string> fit();

		/**
		 * @brief Log a named scalar metric (PyTorch Lightning–style).
		 *
		 * @details
		 * Broadcasts the metric to all loaded plugins via
		 * @ref tmm::plugins::PluginManager::emitMetric.  Plugins that export
		 * @c tmm_on_metric (e.g. console-ui) will receive it immediately.
		 *
		 * @param key    Metric name (e.g. "train_loss", "accuracy").
		 * @param value  Scalar value.
		 */
		void log(std::string_view key, float value);

	private:
		[[nodiscard]] std::expected<float, std::string> runValidation();

		[[nodiscard]] std::string buildFitBeginJson() const;
		[[nodiscard]] std::string buildBatchJson(int64_t epoch, int64_t step, float loss, float lr) const;
		[[nodiscard]] std::string buildEpochJson(const EpochMetrics& m) const;

		conf::TrainingConfig config_;
		plugins::PluginManager& plugins_;
		std::unique_ptr<IModel> model_;
		DatasetFactory trainFactory_;
		DatasetFactory valFactory_;
		std::unique_ptr<IOptimizer> optimizer_;
		std::unique_ptr<ILRScheduler> scheduler_;

		/** @brief Monotonically increasing batch-level step counter.
		 *  Updated by fit(); read by log() for metric step tagging. */
		int64_t globalStep_ = 0;

		std::vector<std::unique_ptr<Callback>> callbacks_;
		/** @brief Snapshot of the latest metric values; read by CallbackMetrics. */
		std::unordered_map<std::string, float> currentMetrics_;
		/** @brief Set to true after config plugins and callbacks have been applied. */
		bool configApplied_ = false;
		/** @brief If set, called between batches; non-null return of true triggers stop. */
		std::function<bool()> stopPredicate_;
	};

} // namespace tmm::trainer

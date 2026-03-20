/**
 * @file trainer.hpp
 * @brief Trainer — orchestrates the full training loop.
 *
 * @details
 * The Trainer owns the model, optimizer, and optional LR scheduler.  It
 * coordinates with the @ref ttm::plugins::PluginManager for lifecycle events
 * and iterates the dataset epoch by epoch via a @ref DatasetFactory.
 *
 * ### Usage
 * @code{.cpp}
 * // Build the trainer, chain optional components, then run.
 * auto result = ttm::trainer::Trainer(cfg, plugins, std::move(model), train_factory)
 *     .optimizer(std::make_unique<MyAdam>(cfg.optimizer.lr))
 *     .scheduler(std::make_unique<MyCosineWarmup>(warmup_steps, total_steps))
 *     .validation(val_factory)
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

#include <ttm/conf/config.hpp>
#include <ttm/datasets/dataset_loader.hpp>
#include <ttm/plugins/plugin_manager.hpp>
#include <ttm/trainer/interfaces.hpp>
#include <ttm/compat/expected.hpp>

#include <functional>
#include <memory>
#include <string>

namespace ttm::trainer {

	/**
	 * @brief Factory function that creates a fresh @ref DatasetIterator.
	 *
	 * Called once per training epoch (and once per validation pass).  The
	 * returned iterator is consumed fully and then discarded.
	 */
	using DatasetFactory = std::function<
		std::expected<std::unique_ptr<ttm::datasets::DatasetIterator>, std::string>()
	>;

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
		Trainer(
			conf::TrainingConfig        config,
			plugins::PluginManager&     plugins,
			std::unique_ptr<IModel>     model,
			DatasetFactory              train
		);

		Trainer(Trainer&&)            = default;
		Trainer& operator=(Trainer&&) = default;
		Trainer(const Trainer&)       = delete;
		Trainer& operator=(const Trainer&) = delete;
		~Trainer()                    = default;

		/// Attach an optimizer.  Without one the model is expected to update itself.
		Trainer& optimizer(std::unique_ptr<IOptimizer>   opt);

		/// Attach an LR scheduler.  Without one the optimizer's initial LR is kept constant.
		Trainer& scheduler(std::unique_ptr<ILRScheduler> sched);

		/// Attach a validation dataset factory (optional).
		Trainer& validation(DatasetFactory val);

		/**
		 * @brief Run the training loop.
		 * @return Final epoch metrics on success, or an error string on failure.
		 */
		[[nodiscard]] std::expected<EpochMetrics, std::string> fit();

	private:
		[[nodiscard]] std::expected<float, std::string> run_validation();

		[[nodiscard]] std::string build_fit_begin_json()                              const;
		[[nodiscard]] std::string build_batch_json(int64_t epoch, int64_t step,
		                                            float loss, float lr)             const;
		[[nodiscard]] std::string build_epoch_json(const EpochMetrics& m)             const;

		conf::TrainingConfig          config_;
		plugins::PluginManager&       plugins_;
		std::unique_ptr<IModel>       model_;
		DatasetFactory                train_factory_;
		DatasetFactory                val_factory_;
		std::unique_ptr<IOptimizer>   optimizer_;
		std::unique_ptr<ILRScheduler> scheduler_;
	};

} // namespace ttm::trainer

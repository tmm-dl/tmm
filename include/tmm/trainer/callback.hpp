/**
 * @file callback.hpp
 * @brief Trainer callback interface — analogous to PyTorch Lightning Callbacks.
 *
 * @details
 * Callbacks are C++ objects attached to the @ref tmm::trainer::Trainer before
 * @c fit() is called.  They receive the same lifecycle events as WASM/native
 * plugins (epoch begin/end, validation, fit begin/end, individual metric logs)
 * but run in-process, with direct access to the Trainer and its current metrics.
 *
 * ### Typical usage
 * @code{.cpp}
 * #include <tmm/trainer/trainer.hpp>
 * #include <tmm/callbacks/early_stopping.hpp>   // from extensions/core
 *
 * trainer.addCallback(
 *     std::make_unique<EarlyStopping>("val_loss", 5));
 * trainer.fit();
 * @endcode
 *
 * ### Adding a custom callback
 * Derive from @ref Callback and override whichever hooks you need.  All hooks
 * have default no-op implementations so you only write what you care about.
 *
 * @code{.cpp}
 * class PrintLoss final : public tmm::trainer::Callback {
 * public:
 *     bool on_epoch_end(Trainer&, int64_t epoch,
 *                       const CallbackMetrics& m) override {
 *         std::cout << "epoch " << epoch
 *                   << "  loss=" << m.get("train_loss") << '\n';
 *         return false;
 *     }
 * };
 * @endcode
 */

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>

namespace tmm::trainer {

	/* Forward declaration — full definition in trainer.hpp.
	 * Callbacks receive a Trainer& so they can call log(), but they do not
	 * need the full definition to compile (they may return before using it). */
	class Trainer;

	/* =========================================================================
	 * CallbackMetrics — read-only snapshot of the trainer's current metrics
	 * ====================================================================== */

	/**
	 * @brief Non-owning, read-only view over the Trainer's current metric values.
	 *
	 * @details
	 * Populated by the Trainer at each lifecycle boundary and passed to callbacks.
	 * At @c on_epoch_end the map contains at least:
	 *   - @c "epoch"           — current 1-based epoch index
	 *   - @c "train_loss"      — epoch-averaged training loss
	 *   - @c "learning_rate"   — current learning rate
	 *   - @c "throughput"      — samples / second
	 *   - @c "val_loss"        — validation loss (if a val factory was set)
	 *
	 * Plus any metric logged by the user or the model via @c Trainer::log().
	 */
	class CallbackMetrics {
	public:
		explicit CallbackMetrics(const std::unordered_map<std::string, float>& m) : m_(m) {}

		/**
		 * @brief Retrieve a metric value by name.
		 * @param key       Metric name (e.g. @c "val_loss").
		 * @param fallback  Returned when the key is absent. Defaults to NaN.
		 * @return Current value, or @p fallback if the metric has not been logged.
		 */
		[[nodiscard]] float get(std::string_view key, float fallback = std::numeric_limits<float>::quiet_NaN()) const {
			const auto it = m_.find(std::string(key));
			return (it != m_.end()) ? it->second : fallback;
		}

		/** @brief Return @c true if the named metric has been logged at least once. */
		[[nodiscard]] bool has(std::string_view key) const { return m_.contains(std::string(key)); }

		/** @brief Direct access to the underlying map for iteration. */
		[[nodiscard]] const std::unordered_map<std::string, float>& all() const { return m_; }

	private:
		const std::unordered_map<std::string, float>& m_;
	};

	/* =========================================================================
	 * Callback — base class
	 * ====================================================================== */

	/**
	 * @brief Base class for all Trainer callbacks.
	 *
	 * @details
	 * All hooks have default no-op implementations.  Derived classes override
	 * only the hooks they need.
	 *
	 * ### Lifecycle order (mirrors PyTorch Lightning):
	 * @code
	 * on_fit_begin
	 *   for each epoch:
	 *     on_epoch_begin
	 *     [ training batches … ]
	 *     on_validation_end   (only when a validation factory is present)
	 *     on_epoch_end  ←  return true here to stop training early
	 * on_fit_end
	 * @endcode
	 */
	class Callback {
	public:
		Callback() = default;
		virtual ~Callback() = default;
		Callback(const Callback&) = delete;
		Callback& operator=(const Callback&) = delete;
		Callback(Callback&&) = default;
		Callback& operator=(Callback&&) = default;

		/**
		 * @brief Called once before the training loop begins.
		 * @param trainer  The owning Trainer (may call @c trainer.log()).
		 * @param metrics  Current metric snapshot (usually empty at this point).
		 */
		virtual void on_fit_begin([[maybe_unused]] Trainer& trainer, [[maybe_unused]] const CallbackMetrics& metrics) {}

		/**
		 * @brief Called once after the training loop finishes (or is stopped early).
		 * @param trainer  The owning Trainer.
		 * @param metrics  Final metric snapshot.
		 */
		virtual void on_fit_end([[maybe_unused]] Trainer& trainer, [[maybe_unused]] const CallbackMetrics& metrics) {}

		/**
		 * @brief Called at the start of each epoch.
		 * @param trainer       The owning Trainer.
		 * @param epoch         1-based epoch number.
		 * @param total_epochs  Total planned epochs.
		 */
		virtual void on_epoch_begin(
				[[maybe_unused]] Trainer& trainer, [[maybe_unused]] int64_t epoch, [[maybe_unused]] int64_t total_epochs
		) {}

		/**
		 * @brief Called at the end of each epoch.
		 *
		 * @param trainer  The owning Trainer.
		 * @param epoch    1-based epoch number.
		 * @param metrics  Epoch-level metric snapshot (see @ref CallbackMetrics).
		 * @return @c true to request early stopping; @c false to continue.
		 *
		 * @note All callbacks are invoked even if an earlier one returns @c true,
		 *       mirroring the existing plugin behaviour.
		 */
		virtual bool on_epoch_end(
				[[maybe_unused]] Trainer& trainer, [[maybe_unused]] int64_t epoch,
				[[maybe_unused]] const CallbackMetrics& metrics
		) {
			return false;
		}

		/**
		 * @brief Called after each validation pass.
		 * @param trainer  The owning Trainer.
		 * @param metrics  Metric snapshot including the freshly computed @c "val_loss".
		 */
		virtual void
		on_validation_end([[maybe_unused]] Trainer& trainer, [[maybe_unused]] const CallbackMetrics& metrics) {}

		/**
		 * @brief Called by @c Trainer::log(key, value) for every scalar metric.
		 *
		 * @details
		 * Invoked both for user-logged metrics (via @c trainer.log()) and for
		 * metrics logged internally by the Trainer itself (train_loss, val_loss,
		 * learning_rate, throughput).
		 *
		 * @param trainer  The owning Trainer.
		 * @param key      Metric name.
		 * @param value    Scalar value.
		 * @param step     Global training step at which the metric was logged.
		 */
		virtual void
		on_log([[maybe_unused]] Trainer& trainer, [[maybe_unused]] std::string_view key, [[maybe_unused]] float value,
			   [[maybe_unused]] int32_t step) {}
	};

} // namespace tmm::trainer

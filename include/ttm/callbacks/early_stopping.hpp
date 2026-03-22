/**
 * @file early_stopping.hpp
 * @brief EarlyStopping callback — stops training when a monitored metric
 *        stops improving.
 *
 * @details
 * Mirrors the PyTorch Lightning EarlyStopping callback.  Attach it to the
 * Trainer before calling fit():
 *
 * @code{.cpp}
 * #include <ttm/callbacks/early_stopping.hpp>
 *
 * trainer
 *     .add_callback(std::make_unique<ttm::callbacks::EarlyStopping>(
 *         "val_loss", 5))   // monitor, patience
 *     .fit();
 * @endcode
 *
 * ### Behaviour
 * At each `on_epoch_end`, the callback reads the named metric from
 * `CallbackMetrics`.  If it fails to improve (by more than @p min_delta)
 * for @p patience consecutive epochs, `on_epoch_end` returns `true` to
 * request early stopping.
 *
 * An epoch is considered an *improvement* when:
 * - `Mode::min` and `current < best - min_delta`
 * - `Mode::max` and `current > best + min_delta`
 *
 * If the monitored metric is absent from the metrics map (e.g. the
 * validation factory was not set) the callback is a no-op for that epoch
 * and emits a warning via `trainer.log()`.
 */

#pragma once

#include <ttm/trainer/callback.hpp>

#include <limits>
#include <string>

namespace ttm::callbacks {

	/**
	 * @brief Stop training when a monitored metric stops improving.
	 */
	class EarlyStopping final : public ttm::trainer::Callback {
	public:
		/** @brief Whether lower or higher metric values represent improvement. */
		enum class Mode { min, max };

		/**
		 * @param monitor    Name of the metric to watch (e.g. @c "val_loss").
		 * @param patience   Number of epochs with no improvement before stopping.
		 * @param mode       @c Mode::min (default) to minimize, @c Mode::max to maximize.
		 * @param min_delta  Minimum absolute change that counts as an improvement.
		 */
		explicit EarlyStopping(std::string monitor, int patience, Mode mode = Mode::min, float min_delta = 0.0f)
				: monitor_(std::move(monitor)), patience_(patience), mode_(mode), min_delta_(min_delta) {}

		void on_fit_begin(
				ttm::trainer::Trainer& /*trainer*/, const ttm::trainer::CallbackMetrics& /*metrics*/
		) override {
			best_ = (mode_ == Mode::min) ? std::numeric_limits<float>::infinity()
										 : -std::numeric_limits<float>::infinity();
			wait_ = 0;
			stopped_ = false;
		}

		bool on_epoch_end(
				ttm::trainer::Trainer& trainer, int64_t /*epoch*/, const ttm::trainer::CallbackMetrics& metrics
		) override {
			if (stopped_)
				return true;

			if (!metrics.has(monitor_)) {
				trainer.log("early_stopping_warn", 0.0f); // no-op sentinel; real warning via emit_log
				return false;
			}

			const float current = metrics.get(monitor_);
			const bool improved =
					(mode_ == Mode::min) ? (current < best_ - min_delta_) : (current > best_ + min_delta_);

			if (improved) {
				best_ = current;
				wait_ = 0;
			} else {
				++wait_;
				if (wait_ >= patience_) {
					stopped_ = true;
					return true;
				}
			}
			return false;
		}

	private:
		std::string monitor_;
		int patience_;
		Mode mode_;
		float min_delta_;

		float best_ = std::numeric_limits<float>::infinity();
		int wait_ = 0;
		bool stopped_ = false;
	};

} // namespace ttm::callbacks

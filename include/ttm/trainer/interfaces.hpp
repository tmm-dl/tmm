/**
 * @file interfaces.hpp
 * @brief Abstract interfaces for model, optimizer, and LR scheduler.
 *
 * @details
 * These interfaces are the extension points that a @ref ttm::trainer::Trainer
 * depends on.  Concrete implementations are provided by the caller — either
 * hand-written for testing or loaded from a TVM module, ONNX model, etc.
 *
 * ### Training vs. inference
 * @ref IModel exposes two separate paths:
 * - @c step()  — forward + backward (accumulates gradients internally).
 * - @c infer() — forward only, no gradient side-effects (used for validation).
 *
 * ### Gradient accumulation
 * The Trainer calls @c IModel::zero_grad() once before the first micro-batch
 * in an accumulation window and @c IModel::step() for each micro-batch.
 * @c IOptimizer::step() is called once at the end of the window.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include <arrow/record_batch.h>

namespace ttm::trainer {

	/* =========================================================================
	 * Batch — one mini-batch of data from a DatasetIterator
	 * ====================================================================== */

	struct Batch {
		std::shared_ptr<arrow::RecordBatch> data;
		int64_t global_index = 0; ///< Monotonically increasing across all epochs.
		int64_t epoch_index  = 0; ///< Batch index within the current epoch.
	};

	/* =========================================================================
	 * StepOutput — result of one forward (+ optional backward) pass
	 * ====================================================================== */

	struct StepOutput {
		float loss = 0.0f;
		// Gradients are accumulated inside IModel; the optimizer reads them directly.
	};

	/* =========================================================================
	 * EpochMetrics — metrics collected at the end of one training epoch
	 * ====================================================================== */

	struct EpochMetrics {
		int64_t epoch        = 0;
		int64_t global_step  = 0; ///< Total optimizer steps taken so far.
		float   train_loss   = 0.0f;
		float   val_loss     = std::numeric_limits<float>::quiet_NaN();
		float   learning_rate = 0.0f;
		double  throughput_samples_per_sec = 0.0;
		std::unordered_map<std::string, float> extras;
	};

	/* =========================================================================
	 * IModel — forward + backward pass
	 * ====================================================================== */

	/**
	 * @brief Abstract model interface.
	 *
	 * Implementations own their parameters and gradient buffers.
	 * The Trainer never touches tensors directly — all compute is behind this
	 * interface, which allows TVM, ONNX-RT, or hand-written kernels to be used
	 * interchangeably.
	 */
	class IModel {
	public:
		IModel()                             = default;
		virtual ~IModel()                    = default;
		IModel(const IModel&)                = delete;
		IModel& operator=(const IModel&)     = delete;
		IModel(IModel&&)                     = default;
		IModel& operator=(IModel&&)          = default;

		[[nodiscard]] virtual std::string_view name() const = 0;

		/**
		 * @brief Forward pass + backward pass (accumulates gradients).
		 * Called once per micro-batch inside an accumulation window.
		 */
		virtual StepOutput step(const Batch& batch)  = 0;

		/**
		 * @brief Forward pass only — no gradient side-effects.
		 * Called during validation and inference.
		 */
		virtual StepOutput infer(const Batch& batch) = 0;

		/// Zero out all accumulated gradients.
		virtual void zero_grad() = 0;
	};

	/* =========================================================================
	 * IOptimizer — gradient-based parameter update
	 * ====================================================================== */

	class IOptimizer {
	public:
		IOptimizer()                                 = default;
		virtual ~IOptimizer()                        = default;
		IOptimizer(const IOptimizer&)                = delete;
		IOptimizer& operator=(const IOptimizer&)     = delete;
		IOptimizer(IOptimizer&&)                     = default;
		IOptimizer& operator=(IOptimizer&&)          = default;

		/// Apply one optimizer step using the gradients currently stored in the model.
		virtual void  step()                         = 0;
		virtual void  zero_grad()                    = 0;

		[[nodiscard]] virtual float learning_rate() const  = 0;
		virtual void  set_learning_rate(float lr)          = 0;
	};

	/* =========================================================================
	 * ILRScheduler — learning-rate schedule
	 * ====================================================================== */

	/**
	 * @brief Maps a global optimizer step index to a learning rate.
	 *
	 * The Trainer calls @c step() after each optimizer step and applies the
	 * returned LR to the optimizer via @c IOptimizer::set_learning_rate().
	 *
	 * @note `global_optimizer_step` is 0-indexed.
	 */
	class ILRScheduler {
	public:
		ILRScheduler()                                     = default;
		virtual ~ILRScheduler()                            = default;
		ILRScheduler(const ILRScheduler&)                  = delete;
		ILRScheduler& operator=(const ILRScheduler&)       = delete;
		ILRScheduler(ILRScheduler&&)                       = default;
		ILRScheduler& operator=(ILRScheduler&&)            = default;

		[[nodiscard]] virtual float step(int64_t global_optimizer_step) = 0;
	};

} // namespace ttm::trainer

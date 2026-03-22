/**
 * @file collator.hpp
 * @brief ICollator — converts an Arrow RecordBatch to a set of DLTensors.
 * @ingroup ttm_model
 */

#pragma once

#include <ttm/compat/expected.hpp>

#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <dlpack/dlpack.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ttm::model {

	/**
	 * @brief A collated mini-batch ready for the model's forward pass.
	 *
	 * @details
	 * Each entry in `inputs` is a DLTensor whose `data` pointer lives inside
	 * `storage` (owned by this struct).  The lifetime of the DLTensors is
	 * therefore tied to the lifetime of this ModelBatch object.  Do not cache
	 * the DLTensor pointers across ModelBatch moves.
	 *
	 * `raw` retains the Arrow RecordBatch from which `inputs` were collated;
	 * it keeps the Arrow memory buffers alive and is useful for debugging.
	 *
	 * @ingroup ttm_model
	 */
	struct ModelBatch {
		std::vector<DLTensor> inputs;			 ///< Collated input tensors.
		std::shared_ptr<arrow::RecordBatch> raw; ///< Original Arrow batch (kept alive).
		/** @brief Backing memory for DLTensors that required a copy (e.g. type conversion). */
		std::vector<std::vector<std::byte>> storage;
		/** @brief Shape arrays owned by this batch; each DLTensor.shape points here. */
		std::vector<std::vector<int64_t>> shapes;
	};

	/**
	 * @brief Converts a preprocessed Arrow RecordBatch to a flat array of DLTensors.
	 *
	 * @details
	 * The default collator (make_default_collator()) handles numeric Arrow
	 * columns by mapping them to DLTensors without copying when the Arrow
	 * buffer layout is compatible (contiguous, non-null array).  A copy is
	 * made only when type conversion is required (e.g. int64 → float32).
	 *
	 * ### Schema compatibility
	 * output_schema() returns the Arrow schema that this collator produces.
	 * ModelPipeline::load() checks that the model's `input_schema_json` is
	 * compatible with the collator's output before training begins.
	 *
	 * ### Custom collators
	 * Implement ICollator to handle:
	 * - Padding variable-length sequences to a fixed length.
	 * - Encoding categorical features.
	 * - Combining multiple columns into a single tensor.
	 *
	 * @ingroup ttm_model
	 */
	class ICollator {
	public:
		ICollator() = default;
		virtual ~ICollator() = default;

		ICollator(const ICollator&) = delete;
		ICollator& operator=(const ICollator&) = delete;
		ICollator(ICollator&&) = delete;
		ICollator& operator=(ICollator&&) = delete;

		/**
		 * @brief Collate a record batch into a ModelBatch.
		 *
		 * @param batch  Input Arrow RecordBatch (after all preprocessors).
		 * @return Collated batch on success, or an error string on failure.
		 */
		[[nodiscard]] virtual std::expected<ModelBatch, std::string> collate(const arrow::RecordBatch& batch) const = 0;

		/**
		 * @brief Arrow schema of the tensors this collator produces.
		 * @details Used by ModelPipeline to validate that the model's expected
		 *          input schema matches the collator's output.
		 */
		[[nodiscard]] virtual const arrow::Schema& output_schema() const = 0;
	};

	/**
	 * @brief Build a default collator from a RecordBatch schema.
	 *
	 * @details
	 * The default collator maps each numeric column (int8, int16, int32,
	 * int64, float32, float64) to a 2-D DLTensor of shape [num_rows, 1].
	 * Non-numeric columns (strings, booleans, etc.) are skipped with a
	 * warning.
	 *
	 * @param schema  Arrow schema of the batches this collator will receive.
	 * @return Fully configured ICollator instance.
	 */
	[[nodiscard]] std::unique_ptr<ICollator> make_default_collator(const arrow::Schema& schema);

} // namespace ttm::model

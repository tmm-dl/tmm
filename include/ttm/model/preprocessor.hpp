/**
 * @file preprocessor.hpp
 * @brief IPreprocessor — Arrow RecordBatch → Arrow RecordBatch transform.
 * @ingroup ttm_model
 */

#pragma once

#include <ttm/compat/expected.hpp>

#include <arrow/record_batch.h>

#include <memory>
#include <string>
#include <string_view>

namespace ttm::model {

	/**
	 * @brief Transforms an Arrow RecordBatch before collation.
	 *
	 * @details
	 * Preprocessors run in a fixed order configured in the training YAML:
	 *
	 * @code{.yaml}
	 * preprocessors:
	 *   - type: bpe-tokenize
	 *     config: '{"vocab": "vocab.json"}'
	 *   - type: truncate
	 *     config: '{"max_length": 512}'
	 * @endcode
	 *
	 * Each preprocessor receives the RecordBatch output by the previous one
	 * (or the raw dataset batch for the first preprocessor) and must return a
	 * new RecordBatch.  The batch passed to the model is the output of the
	 * last preprocessor.
	 *
	 * ### Implementing a preprocessor
	 * Native C++ preprocessors subclass IPreprocessor directly.  WASM / shared-
	 * library plugins register a #ttm_transform_vtable and receive a C++ adapter
	 * (ttm::plugins::ITransform) that wraps the vtable.
	 *
	 * @see ttm::plugins::ITransform  C ABI adapter around this interface
	 * @ingroup ttm_model
	 */
	class IPreprocessor {
	public:
		IPreprocessor() = default;
		virtual ~IPreprocessor() = default;

		IPreprocessor(const IPreprocessor&) = delete;
		IPreprocessor& operator=(const IPreprocessor&) = delete;
		IPreprocessor(IPreprocessor&&) = delete;
		IPreprocessor& operator=(IPreprocessor&&) = delete;

		/**
		 * @brief Canonical name used in the `preprocessors[].type` config field.
		 * @return Non-owning string view into plugin-owned storage.
		 */
		[[nodiscard]] virtual std::string_view name() const = 0;

		/**
		 * @brief Transform a record batch.
		 *
		 * @param batch  Input batch from the dataset iterator (or previous
		 *               preprocessor).  The reference is valid only for the
		 *               duration of this call.
		 * @return Transformed batch on success, or an error string on failure.
		 *         Returning an error aborts the current training step.
		 */
		[[nodiscard]] virtual std::expected<std::shared_ptr<arrow::RecordBatch>, std::string>
		apply(const arrow::RecordBatch& batch) const = 0;
	};

} // namespace ttm::model

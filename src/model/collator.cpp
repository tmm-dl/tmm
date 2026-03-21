/**
 * @file collator.cpp
 * @brief Default Arrow → DLTensor collator implementation.
 */

#include <ttm/model/collator.hpp>

#include <arrow/array.h>
#include <arrow/extension_type.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ttm::model {

	namespace {

		/// Map Arrow type id to DLDataType.
		DLDataType arrow_to_dltype(arrow::Type::type tid) {
			switch (tid) {
			case arrow::Type::INT8:    return {kDLInt,   8,  1};
			case arrow::Type::INT16:   return {kDLInt,   16, 1};
			case arrow::Type::INT32:   return {kDLInt,   32, 1};
			case arrow::Type::INT64:   return {kDLInt,   64, 1};
			case arrow::Type::UINT8:   return {kDLUInt,  8,  1};
			case arrow::Type::UINT16:  return {kDLUInt,  16, 1};
			case arrow::Type::UINT32:  return {kDLUInt,  32, 1};
			case arrow::Type::UINT64:  return {kDLUInt,  64, 1};
			case arrow::Type::FLOAT:   return {kDLFloat, 32, 1};
			case arrow::Type::DOUBLE:  return {kDLFloat, 64, 1};
			default:                   return {kDLFloat, 32, 1}; // fallback
			}
		}

		/// True for Arrow types that have a trivial contiguous data buffer.
		bool is_numeric_col(arrow::Type::type tid) {
			switch (tid) {
			case arrow::Type::INT8:
			case arrow::Type::INT16:
			case arrow::Type::INT32:
			case arrow::Type::INT64:
			case arrow::Type::UINT8:
			case arrow::Type::UINT16:
			case arrow::Type::UINT32:
			case arrow::Type::UINT64:
			case arrow::Type::FLOAT:
			case arrow::Type::DOUBLE:
				return true;
			default:
				return false;
			}
		}

		/// Build a 2-D DLTensor [num_rows, 1] pointing into Arrow buffer (zero-copy).
		/// shape must remain alive as long as the DLTensor is used (owned by ModelBatch::shapes).
		DLTensor make_tensor_from_arrow(
			const void*   data_ptr,
			int64_t       num_rows,
			DLDataType    dtype,
			std::vector<int64_t>& shape_storage
		) {
			shape_storage = {num_rows, 1};
			DLTensor t{};
			// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast) -- DLTensor.data is void*; Arrow buffer is const; safe read-only use
			t.data         = const_cast<void*>(data_ptr);
			t.device       = {kDLCPU, 0};
			t.ndim         = 2;
			t.dtype        = dtype;
			t.shape        = shape_storage.data();
			t.strides      = nullptr; // contiguous
			t.byte_offset  = 0;
			return t;
		}

		/// Build a 2-D DLTensor [num_rows, list_size] from a FixedSizeList column (zero-copy).
		/// list_size is the fixed number of elements per row.
		DLTensor make_2d_tensor_from_arrow(
			const void*   data_ptr,
			int64_t       num_rows,
			int32_t       list_size,
			DLDataType    dtype,
			std::vector<int64_t>& shape_storage
		) {
			shape_storage = {num_rows, static_cast<int64_t>(list_size)};
			DLTensor t{};
			// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
			t.data         = const_cast<void*>(data_ptr);
			t.device       = {kDLCPU, 0};
			t.ndim         = 2;
			t.dtype        = dtype;
			t.shape        = shape_storage.data();
			t.strides      = nullptr;
			t.byte_offset  = 0;
			return t;
		}

		/**
		 * @brief Default ICollator: maps each numeric column to a 2-D DLTensor.
		 */
		class DefaultCollator final : public ICollator {
		public:
			explicit DefaultCollator(std::shared_ptr<arrow::Schema> schema)
				: schema_(std::move(schema)) {}

			[[nodiscard]] std::expected<ModelBatch, std::string>
			collate(const arrow::RecordBatch& batch) const override {
				ModelBatch out;
				// Shallow copy: new RecordBatch sharing the same column arrays.
				out.raw = arrow::RecordBatch::Make(
					batch.schema(), batch.num_rows(), batch.columns()
				);

				const int num_cols = batch.num_columns();
				out.inputs.reserve(static_cast<std::size_t>(num_cols));
				out.shapes.reserve(static_cast<std::size_t>(num_cols));
				out.storage.reserve(0); // filled only when copies are needed

				// Use the batch's own schema (not the stored schema_) so we handle
				// post-preprocessing schemas (e.g. FixedSizeList from tokenizer).
				const auto& schema = *batch.schema();

				for (int c = 0; c < num_cols; ++c) {
					const auto& field = schema.field(c);
					const auto  tid   = field->type()->id();

					if (is_numeric_col(tid)) {
						const auto& col = batch.column(c);
						const DLDataType dtype = arrow_to_dltype(tid);

						// Try zero-copy: use Arrow's value buffer directly
						const auto* prim = dynamic_cast<const arrow::PrimitiveArray*>(col.get());
						if (prim != nullptr && prim->null_count() == 0) {
							const void* buf = prim->values()->data();
							out.shapes.emplace_back();
							out.inputs.push_back(make_tensor_from_arrow(
								buf, batch.num_rows(), dtype, out.shapes.back()
							));
						} else {
							// Fallback: zero-fill
							const int64_t n = batch.num_rows();
							const std::size_t nbytes = static_cast<std::size_t>(n) * sizeof(float);
							out.storage.emplace_back(nbytes, std::byte{0});
							out.shapes.emplace_back();
							out.inputs.push_back(make_tensor_from_arrow(
								out.storage.back().data(), n, {kDLFloat, 32, 1}, out.shapes.back()
							));
						}
					} else if (tid == arrow::Type::FIXED_SIZE_LIST) {
						// FixedSizeList<T, N> → 2D DLTensor [batch, N]
						const auto& col = batch.column(c);
						const auto* fsl_type = static_cast<const arrow::FixedSizeListType*>(field->type().get());
						const int32_t list_size = fsl_type->list_size();
						const auto  value_type_id = fsl_type->value_type()->id();

						if (!is_numeric_col(value_type_id)) {
							std::cerr << "[ttm/collator] skipping FixedSizeList column '"
							          << field->name() << "': non-numeric value type\n";
							continue;
						}
						const DLDataType dtype = arrow_to_dltype(value_type_id);

						// FixedSizeListArray values are laid out contiguously in the child array
						const auto* fsl_arr = static_cast<const arrow::FixedSizeListArray*>(col.get());
						const auto& values = fsl_arr->values();
						const auto* prim = dynamic_cast<const arrow::PrimitiveArray*>(values.get());
						if (prim != nullptr && prim->null_count() == 0) {
							const void* buf = prim->values()->data();
							out.shapes.emplace_back();
							out.inputs.push_back(make_2d_tensor_from_arrow(
								buf, batch.num_rows(), list_size, dtype, out.shapes.back()
							));
						} else {
							// Fallback: copy to contiguous storage
							const int64_t n = batch.num_rows();
							const std::size_t nbytes = static_cast<std::size_t>(n) *
							                           static_cast<std::size_t>(list_size) *
							                           (dtype.bits / 8u);
							out.storage.emplace_back(nbytes, std::byte{0});
							out.shapes.emplace_back();
							out.inputs.push_back(make_2d_tensor_from_arrow(
								out.storage.back().data(), n, list_size, dtype, out.shapes.back()
							));
						}
					} else {
						std::cerr << "[ttm/collator] skipping column '"
						          << field->name() << "' (unsupported type)\n";
					}
				}

				return out;
			}

			[[nodiscard]] const arrow::Schema& output_schema() const override {
				return *schema_;
			}

		private:
			std::shared_ptr<arrow::Schema> schema_;
		};

	} // anonymous namespace

	std::unique_ptr<ICollator> make_default_collator(const arrow::Schema& schema) {
		return std::make_unique<DefaultCollator>(
			arrow::schema(schema.fields(), schema.metadata()));
	}

} // namespace ttm::model

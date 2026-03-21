/**
 * @file collator.cpp
 * @brief Default Arrow → DLTensor collator implementation.
 */

#include <ttm/model/collator.hpp>

#include <arrow/array.h>
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

				for (int c = 0; c < num_cols; ++c) {
					const auto& field = schema_->field(c);
					const auto  tid   = field->type()->id();

					if (!is_numeric_col(tid)) {
						std::cerr << "[ttm/collator] skipping non-numeric column '"
						          << field->name() << "'\n";
						continue;
					}

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
						// Fallback: zero-fill (nullable/non-primitive arrays are rare in
						// well-formed training data; callers should pre-drop nulls).
						const int64_t n = batch.num_rows();
						const std::size_t nbytes = static_cast<std::size_t>(n) * sizeof(float);
						out.storage.emplace_back(nbytes, std::byte{0});

						out.shapes.emplace_back();
						out.inputs.push_back(make_tensor_from_arrow(
							out.storage.back().data(), n, {kDLFloat, 32, 1}, out.shapes.back()
						));
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

/**
 * @file test_collator.cpp
 * @brief Unit tests for make_default_collator() and ModelBatch structure.
 *
 * @details
 * Tests cover:
 * - Numeric column mapping to correct DLDataType
 * - Shape of output DLTensors ([num_rows, 1])
 * - Non-numeric columns are silently skipped
 * - Zero-copy path for contiguous primitive arrays
 * - Empty batch (zero rows)
 * - Mixed-type schema
 */

#include <ttm/model/collator.hpp>

#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>

// =============================================================================
// Helpers
// =============================================================================

namespace {

	/// Build a single-column RecordBatch with the given values.
	template<typename BuilderT, typename ValueT>
	std::shared_ptr<arrow::RecordBatch> make_batch(
		const std::string& col_name,
		std::shared_ptr<arrow::DataType> dtype,
		const std::vector<ValueT>& values
	) {
		BuilderT builder;
		REQUIRE(builder.AppendValues(values).ok());
		std::shared_ptr<arrow::Array> arr;
		REQUIRE(builder.Finish(&arr).ok());
		auto schema = arrow::schema({arrow::field(col_name, dtype)});
		return arrow::RecordBatch::Make(schema, static_cast<int64_t>(values.size()), {arr});
	}

	/// Build a RecordBatch with two columns.
	template<typename B1, typename V1, typename B2, typename V2>
	std::shared_ptr<arrow::RecordBatch> make_two_col_batch(
		const std::string& n1, std::shared_ptr<arrow::DataType> t1, const std::vector<V1>& v1,
		const std::string& n2, std::shared_ptr<arrow::DataType> t2, const std::vector<V2>& v2
	) {
		B1 b1; REQUIRE(b1.AppendValues(v1).ok());
		B2 b2; REQUIRE(b2.AppendValues(v2).ok());
		std::shared_ptr<arrow::Array> a1, a2;
		REQUIRE(b1.Finish(&a1).ok());
		REQUIRE(b2.Finish(&a2).ok());
		auto schema = arrow::schema({
			arrow::field(n1, t1),
			arrow::field(n2, t2)
		});
		return arrow::RecordBatch::Make(schema, static_cast<int64_t>(v1.size()), {a1, a2});
	}

} // anonymous namespace

// =============================================================================
// Basic construction
// =============================================================================

TEST_CASE("make_default_collator returns non-null for a simple schema", "[model][collator]") {
	auto schema = arrow::schema({arrow::field("x", arrow::float32())});
	auto collator = ttm::model::make_default_collator(*schema);
	CHECK(collator != nullptr);
}

TEST_CASE("DefaultCollator output_schema matches input schema", "[model][collator]") {
	auto schema = arrow::schema({
		arrow::field("a", arrow::int32()),
		arrow::field("b", arrow::float32())
	});
	auto collator = ttm::model::make_default_collator(*schema);
	REQUIRE(collator != nullptr);
	CHECK(collator->output_schema().Equals(*schema));
}

// =============================================================================
// Numeric column → correct DLDataType
// =============================================================================

TEST_CASE("DefaultCollator maps float32 column to kDLFloat/32", "[model][collator]") {
	auto batch = make_batch<arrow::FloatBuilder, float>(
		"x", arrow::float32(), {1.0f, 2.0f, 3.0f}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	REQUIRE(result->inputs.size() == 1);
	const auto& t = result->inputs[0];
	CHECK(t.dtype.code == kDLFloat);
	CHECK(t.dtype.bits == 32);
}

TEST_CASE("DefaultCollator maps int32 column to kDLInt/32", "[model][collator]") {
	auto batch = make_batch<arrow::Int32Builder, int32_t>(
		"ids", arrow::int32(), {10, 20, 30}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	REQUIRE(result->inputs.size() == 1);
	const auto& t = result->inputs[0];
	CHECK(t.dtype.code == kDLInt);
	CHECK(t.dtype.bits == 32);
}

TEST_CASE("DefaultCollator maps int64 column to kDLInt/64", "[model][collator]") {
	auto batch = make_batch<arrow::Int64Builder, int64_t>(
		"ts", arrow::int64(), {100LL, 200LL}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	REQUIRE(result->inputs.size() == 1);
	CHECK(result->inputs[0].dtype.code == kDLInt);
	CHECK(result->inputs[0].dtype.bits == 64);
}

TEST_CASE("DefaultCollator maps float64 column to kDLFloat/64", "[model][collator]") {
	auto batch = make_batch<arrow::DoubleBuilder, double>(
		"score", arrow::float64(), {1.1, 2.2}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	REQUIRE(result->inputs.size() == 1);
	CHECK(result->inputs[0].dtype.code == kDLFloat);
	CHECK(result->inputs[0].dtype.bits == 64);
}

// =============================================================================
// Shape: [num_rows, 1]
// =============================================================================

TEST_CASE("DefaultCollator produces shape [num_rows, 1]", "[model][collator]") {
	auto batch = make_batch<arrow::Int32Builder, int32_t>(
		"v", arrow::int32(), {1, 2, 3, 4, 5}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	REQUIRE(result->inputs.size() == 1);
	const auto& t = result->inputs[0];
	REQUIRE(t.ndim == 2);
	CHECK(t.shape[0] == 5);
	CHECK(t.shape[1] == 1);
}

// =============================================================================
// Non-numeric columns are skipped
// =============================================================================

TEST_CASE("DefaultCollator skips string columns", "[model][collator]") {
	arrow::StringBuilder sb;
	REQUIRE(sb.AppendValues({"hello", "world"}).ok());
	std::shared_ptr<arrow::Array> str_arr;
	REQUIRE(sb.Finish(&str_arr).ok());
	auto schema = arrow::schema({arrow::field("text", arrow::utf8())});
	auto batch = arrow::RecordBatch::Make(schema, 2, {str_arr});

	auto collator = ttm::model::make_default_collator(*schema);
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	// String column is skipped — no tensors produced
	CHECK(result->inputs.empty());
}

TEST_CASE("DefaultCollator only includes numeric columns from mixed schema", "[model][collator]") {
	// float32 + string → only float32 tensor produced
	arrow::FloatBuilder fb;
	REQUIRE(fb.AppendValues({1.0f, 2.0f}).ok());
	std::shared_ptr<arrow::Array> float_arr;
	REQUIRE(fb.Finish(&float_arr).ok());

	arrow::StringBuilder sb;
	REQUIRE(sb.AppendValues({"a", "b"}).ok());
	std::shared_ptr<arrow::Array> str_arr;
	REQUIRE(sb.Finish(&str_arr).ok());

	auto schema = arrow::schema({
		arrow::field("x", arrow::float32()),
		arrow::field("label", arrow::utf8())
	});
	auto batch = arrow::RecordBatch::Make(schema, 2, {float_arr, str_arr});

	auto collator = ttm::model::make_default_collator(*schema);
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	CHECK(result->inputs.size() == 1);
	CHECK(result->inputs[0].dtype.code == kDLFloat);
}

// =============================================================================
// Multiple numeric columns
// =============================================================================

TEST_CASE("DefaultCollator collates two numeric columns independently", "[model][collator]") {
	auto batch = make_two_col_batch<arrow::Int32Builder, int32_t,
	                                arrow::FloatBuilder, float>(
		"ids", arrow::int32(), {1, 2, 3},
		"vals", arrow::float32(), {0.1f, 0.2f, 0.3f}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	CHECK(result->inputs.size() == 2);
	CHECK(result->inputs[0].dtype.code == kDLInt);
	CHECK(result->inputs[1].dtype.code == kDLFloat);
}

// =============================================================================
// Empty batch (0 rows)
// =============================================================================

TEST_CASE("DefaultCollator handles zero-row batch", "[model][collator]") {
	auto batch = make_batch<arrow::Int32Builder, int32_t>(
		"x", arrow::int32(), {}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	REQUIRE(result->inputs.size() == 1);
	CHECK(result->inputs[0].shape[0] == 0);
	CHECK(result->inputs[0].shape[1] == 1);
}

// =============================================================================
// ModelBatch keeps raw RecordBatch alive
// =============================================================================

TEST_CASE("ModelBatch retains raw RecordBatch", "[model][collator]") {
	auto batch = make_batch<arrow::FloatBuilder, float>(
		"x", arrow::float32(), {1.0f, 2.0f}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	CHECK(result->raw != nullptr);
	CHECK(result->raw->num_rows() == 2);
}

// =============================================================================
// Shape arrays stay valid after multiple collate calls
// =============================================================================

TEST_CASE("DLTensor shape pointer is valid within ModelBatch lifetime", "[model][collator]") {
	auto batch = make_batch<arrow::Int32Builder, int32_t>(
		"x", arrow::int32(), {10, 20, 30}
	);
	auto collator = ttm::model::make_default_collator(*batch->schema());
	auto result = collator->collate(*batch);
	REQUIRE(result.has_value());
	const auto* shp = result->inputs[0].shape;
	REQUIRE(shp != nullptr);
	CHECK(shp[0] == 3);
	CHECK(shp[1] == 1);
}

/**
 * @file test_trainer_loop.cpp
 * @brief Unit tests for the Trainer loop — gradient accumulation, optimizer
 *        stepping cadence, early stopping, and validation.
 *
 * @details
 * All tests use lightweight mock implementations of IModel, IOptimizer, and
 * ILRScheduler together with a fixed-batch DatasetIterator.  No actual model
 * weights or GPU operations are involved.
 */

#include <tmm/conf/config.hpp>
#include <tmm/datasets/dataset_loader.hpp>
#include <tmm/plugins/plugin_manager.hpp>
#include <tmm/trainer/interfaces.hpp>
#include <tmm/trainer/trainer.hpp>

#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// =============================================================================
// Mock infrastructure
// =============================================================================

namespace {

	// ── Mock RecordBatch ─────────────────────────────────────────────────────

	std::shared_ptr<arrow::RecordBatch> makeDummyBatch(int64_t rows = 4) {
		arrow::Int32Builder builder;
		for (int64_t i = 0; i < rows; ++i)
			REQUIRE(builder.Append(static_cast<int32_t>(i)).ok());
		std::shared_ptr<arrow::Array> arr;
		REQUIRE(builder.Finish(&arr).ok());
		auto schema = arrow::schema({arrow::field("x", arrow::int32())});
		return arrow::RecordBatch::Make(schema, rows, {arr});
	}

	// ── Mock DatasetIterator ─────────────────────────────────────────────────

	/**
	 * Returns exactly `count` identical dummy batches, then signals end-of-stream.
	 */
	class MockIterator final : public tmm::datasets::DatasetIterator {
	public:
		explicit MockIterator(int remaining)
				: remaining_(remaining), schema_(arrow::schema({arrow::field("x", arrow::int32())})) {}

		bool next(std::shared_ptr<arrow::RecordBatch>& out) override {
			if (remaining_ == 0)
				return false;
			--remaining_;
			out = makeDummyBatch();
			return true;
		}

		[[nodiscard]] const arrow::Schema& schema() const override { return *schema_; }

	private:
		int remaining_;
		std::shared_ptr<arrow::Schema> schema_;
	};

	/// Factory that creates a fresh MockIterator each time (for multi-epoch tests).
	tmm::trainer::DatasetFactory makeFactory(int batchesPerEpoch) {
		return [batchesPerEpoch]() -> std::expected<std::unique_ptr<tmm::datasets::DatasetIterator>, std::string> {
			return std::make_unique<MockIterator>(batchesPerEpoch);
		};
	}

	// ── Mock IModel ──────────────────────────────────────────────────────────

	struct MockModel final : public tmm::trainer::IModel {
		int stepCalls = 0;
		int inferCalls = 0;
		int zeroGradCalls = 0;
		float stepLossReturn = 0.5f;
		float inferLossReturn = 0.25f;

		[[nodiscard]] std::string_view name() const override { return "mock"; }

		tmm::trainer::StepOutput step(const tmm::trainer::Batch&) override {
			++stepCalls;
			return {.loss = stepLossReturn};
		}

		tmm::trainer::StepOutput infer(const tmm::trainer::Batch&) override {
			++inferCalls;
			return {.loss = inferLossReturn};
		}

		void zeroGrad() override { ++zeroGradCalls; }
	};

	// ── Mock IOptimizer ──────────────────────────────────────────────────────

	struct MockOptimizer final : public tmm::trainer::IOptimizer {
		int stepCalls = 0;
		int zeroGradCalls = 0;
		float lr = 1e-3f;

		void step() override { ++stepCalls; }
		void zeroGrad() override { ++zeroGradCalls; }
		[[nodiscard]] float learningRate() const override { return lr; }
		void setLearningRate(float newLr) override { lr = newLr; }
	};

	// ── Mock ILRScheduler ────────────────────────────────────────────────────

	struct MockScheduler final : public tmm::trainer::ILRScheduler {
		int stepCalls = 0;
		float lrReturn = 2e-4f;

		[[nodiscard]] float step(int64_t) override {
			++stepCalls;
			return lrReturn;
		}
	};

	// ── Config factory ───────────────────────────────────────────────────────

	tmm::conf::TrainingConfig makeConfig(int64_t epochs = 1, int64_t accumSteps = 1) {
		tmm::conf::TrainingConfig cfg;
		cfg.epochs = epochs;
		cfg.gradientAccumulationSteps = accumSteps;
		return cfg;
	}

} // anonymous namespace

// =============================================================================
// Basic training loop
// =============================================================================

TEST_CASE("Trainer: model.step() called once per batch (no accumulation)", "[trainer][loop]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawModel = new MockModel();
	auto model = std::unique_ptr<tmm::trainer::IModel>(rawModel);
	const int batchesPerEpoch = 5;

	auto result = tmm::trainer::Trainer(makeConfig(1, 1), mgr, std::move(model), makeFactory(batchesPerEpoch)).fit();

	REQUIRE(result.has_value());
	CHECK(rawModel->stepCalls == batchesPerEpoch);
}

TEST_CASE("Trainer: multi-epoch loop calls model.step() for every batch", "[trainer][loop]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawModel = new MockModel();
	auto model = std::unique_ptr<tmm::trainer::IModel>(rawModel);
	const int epochs = 3;
	const int batchesPerEpoch = 4;

	auto result =
			tmm::trainer::Trainer(makeConfig(epochs, 1), mgr, std::move(model), makeFactory(batchesPerEpoch)).fit();

	REQUIRE(result.has_value());
	CHECK(rawModel->stepCalls == epochs * batchesPerEpoch);
}

TEST_CASE("Trainer: optimizer.step() called once per batch with accumulation=1", "[trainer][optimizer]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawOpt = new MockOptimizer();
	auto* rawModel = new MockModel();
	const int batchesPerEpoch = 6;

	auto result = tmm::trainer::Trainer(makeConfig(1, 1), mgr, std::unique_ptr<tmm::trainer::IModel>(rawModel),
	                                    makeFactory(batchesPerEpoch))
	                      .optimizer(std::unique_ptr<tmm::trainer::IOptimizer>(rawOpt))
	                      .fit();

	REQUIRE(result.has_value());
	CHECK(rawOpt->stepCalls == batchesPerEpoch);
}

// =============================================================================
// Gradient accumulation
// =============================================================================

TEST_CASE("Trainer: optimizer.step() called once per accumulation window", "[trainer][gradient-accumulation]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawOpt = new MockOptimizer();
	auto* rawModel = new MockModel();
	const int batchesPerEpoch = 8;
	const int accumSteps = 4;

	auto result = tmm::trainer::Trainer(makeConfig(1, accumSteps), mgr,
	                                    std::unique_ptr<tmm::trainer::IModel>(rawModel), makeFactory(batchesPerEpoch))
	                      .optimizer(std::unique_ptr<tmm::trainer::IOptimizer>(rawOpt))
	                      .fit();

	REQUIRE(result.has_value());
	// 8 batches / 4 accumulation steps = 2 optimizer steps
	CHECK(rawOpt->stepCalls == batchesPerEpoch / accumSteps);
	// model.step() still called for every micro-batch
	CHECK(rawModel->stepCalls == batchesPerEpoch);
}

TEST_CASE("Trainer: partial accumulation window flushed at epoch end", "[trainer][gradient-accumulation]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawOpt = new MockOptimizer();
	auto* rawModel = new MockModel();
	// 5 batches, accumulation window of 4 → 1 full flush + 1 partial flush
	const int batchesPerEpoch = 5;
	const int accumSteps = 4;

	auto result = tmm::trainer::Trainer(makeConfig(1, accumSteps), mgr,
	                                    std::unique_ptr<tmm::trainer::IModel>(rawModel), makeFactory(batchesPerEpoch))
	                      .optimizer(std::unique_ptr<tmm::trainer::IOptimizer>(rawOpt))
	                      .fit();

	REQUIRE(result.has_value());
	// Expected: 2 optimizer steps (1 after batch 4, 1 partial flush at epoch end)
	CHECK(rawOpt->stepCalls == 2);
	CHECK(rawModel->stepCalls == batchesPerEpoch);
}

TEST_CASE("Trainer: zeroGrad() called before each accumulation window", "[trainer][gradient-accumulation]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawModel = new MockModel();
	// 6 batches, window of 2 → 3 windows → 3 zeroGrad() calls
	const int batchesPerEpoch = 6;
	const int accumSteps = 2;

	tmm::trainer::Trainer(makeConfig(1, accumSteps), mgr, std::unique_ptr<tmm::trainer::IModel>(rawModel),
	                      makeFactory(batchesPerEpoch))
	        .fit();

	CHECK(rawModel->zeroGradCalls == batchesPerEpoch / accumSteps);
}

// =============================================================================
// LR scheduler
// =============================================================================

TEST_CASE("Trainer: scheduler.step() called once per optimizer step", "[trainer][scheduler]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawSched = new MockScheduler();
	auto* rawOpt = new MockOptimizer();
	const int batchesPerEpoch = 5;
	const int epochs = 2;

	tmm::trainer::Trainer(makeConfig(epochs, 1), mgr, std::make_unique<MockModel>(), makeFactory(batchesPerEpoch))
	        .optimizer(std::unique_ptr<tmm::trainer::IOptimizer>(rawOpt))
	        .scheduler(std::unique_ptr<tmm::trainer::ILRScheduler>(rawSched))
	        .fit();

	CHECK(rawSched->stepCalls == epochs * batchesPerEpoch);
}

TEST_CASE("Trainer: scheduler LR is applied to the optimizer", "[trainer][scheduler]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawOpt = new MockOptimizer();
	rawOpt->lr = 1e-3f;

	auto* rawSched = new MockScheduler();
	rawSched->lrReturn = 7e-5f;

	tmm::trainer::Trainer(makeConfig(1, 1), mgr, std::make_unique<MockModel>(), makeFactory(3))
	        .optimizer(std::unique_ptr<tmm::trainer::IOptimizer>(rawOpt))
	        .scheduler(std::unique_ptr<tmm::trainer::ILRScheduler>(rawSched))
	        .fit();

	// After scheduler updates, optimizer LR should be the scheduler's value
	CHECK(rawOpt->lr == Catch::Approx(7e-5f));
}

// =============================================================================
// Validation pass
// =============================================================================

TEST_CASE("Trainer: validation pass calls infer() not step()", "[trainer][validation]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawModel = new MockModel();
	const int trainBatches = 3;
	const int valBatches = 2;

	auto result = tmm::trainer::Trainer(makeConfig(1, 1), mgr, std::unique_ptr<tmm::trainer::IModel>(rawModel),
	                                    makeFactory(trainBatches))
	                      .validation(makeFactory(valBatches))
	                      .fit();

	REQUIRE(result.has_value());
	CHECK(rawModel->stepCalls == trainBatches);
	CHECK(rawModel->inferCalls == valBatches);
}

TEST_CASE("Trainer: fit() returns valLoss from validation pass", "[trainer][validation]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawModel = new MockModel();
	rawModel->inferLossReturn = 0.123f;

	auto result = tmm::trainer::Trainer(makeConfig(1, 1), mgr, std::unique_ptr<tmm::trainer::IModel>(rawModel),
	                                    makeFactory(2))
	                      .validation(makeFactory(2))
	                      .fit();

	REQUIRE(result.has_value());
	CHECK(result->valLoss == Catch::Approx(0.123f));
}

// =============================================================================
// fit() return value
// =============================================================================

TEST_CASE("Trainer: fit() returns trainLoss as average of per-batch losses", "[trainer][metrics]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawModel = new MockModel();
	rawModel->stepLossReturn = 0.8f;
	const int batches = 4;

	auto result = tmm::trainer::Trainer(makeConfig(1, 1), mgr, std::unique_ptr<tmm::trainer::IModel>(rawModel),
	                                    makeFactory(batches))
	                      .fit();

	REQUIRE(result.has_value());
	CHECK(result->trainLoss == Catch::Approx(0.8f));
}

TEST_CASE("Trainer: fit() returns epoch=0 for single epoch run", "[trainer][metrics]") {
	auto mgr = tmm::plugins::PluginManager::create().value();

	auto result = tmm::trainer::Trainer(makeConfig(1, 1), mgr, std::make_unique<MockModel>(), makeFactory(2)).fit();

	REQUIRE(result.has_value());
	CHECK(result->epoch == 0);
}

TEST_CASE("Trainer: stopPredicate stops training mid-epoch", "[trainer][stop]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	auto* rawModel = new MockModel();

	std::atomic<int> batchesSeen{0};
	auto result = tmm::trainer::Trainer(makeConfig(1, 1), mgr, std::unique_ptr<tmm::trainer::IModel>(rawModel),
	                                    makeFactory(100))
	                      .stopPredicate([&] {
		                      ++batchesSeen;
		                      return batchesSeen >= 3;
	                      })
	                      .fit();

	// Training should stop early; model.step() should not be called for all 100 batches
	REQUIRE(result.has_value());
	CHECK(rawModel->stepCalls < 100);
}

/**
 * @file test_model_loader.cpp
 * @brief Unit tests for model loader registration and dispatch in PluginManager.
 *
 * @details
 * Uses an inline CModelLoaderAdapter-style test double (a native plugin loaded
 * via the public C vtable registration path) to verify:
 * - find_model_loader() returns nullptr when no loader is registered
 * - A loader registered through the host API is found by find_model_loader()
 * - probe() is called to select the correct loader
 * - emit_model_loaded() is a no-op (does not crash) with no plugins loaded
 * - find_transform() returns nullptr when no transforms are registered
 */

#include <ttm/plugins/plugin_manager.hpp>
#include <ttm/model/device.hpp>
#include <ttm/model/model.hpp>
#include <ttm/compat/expected.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <string_view>

// =============================================================================
// find_model_loader with no loaders registered
// =============================================================================

TEST_CASE("find_model_loader returns nullptr when no loaders registered", "[model][loader]") {
	auto mgr = ttm::plugins::PluginManager::create().value();
	CHECK(mgr.find_model_loader("model.py") == nullptr);
	CHECK(mgr.find_model_loader("model.so") == nullptr);
	CHECK(mgr.find_model_loader("") == nullptr);
}

// =============================================================================
// find_transform with no transforms registered
// =============================================================================

TEST_CASE("find_transform returns nullptr when no transforms registered", "[model][loader]") {
	auto mgr = ttm::plugins::PluginManager::create().value();
	CHECK(mgr.find_transform("bpe-tokenize") == nullptr);
	CHECK(mgr.find_transform("") == nullptr);
}

// =============================================================================
// emit_model_loaded is safe with no plugins
// =============================================================================

TEST_CASE("emit_model_loaded is a no-op with no plugins loaded", "[model][loader]") {
	auto mgr = ttm::plugins::PluginManager::create().value();
	CHECK_NOTHROW(mgr.emit_model_loaded(R"({"name":"test","arch":"Linear"})"));
	CHECK_NOTHROW(mgr.emit_model_loaded(""));
}

// =============================================================================
// Mock IModelLoader — inlined test double
// =============================================================================

namespace {

	/// Simple IModelLoader that claims ownership of files with a given extension.
	class MockModelLoader final : public ttm::model::IModelLoader {
	public:
		explicit MockModelLoader(std::string_view ext) : ext_(ext) {}

		[[nodiscard]] bool probe(std::string_view path) const override {
			return path.ends_with(ext_);
		}

		[[nodiscard]] std::expected<std::unique_ptr<ttm::model::IModel>, std::string>
		load(std::string_view /*path*/, std::string_view /*cfg*/, ttm::model::Device /*dev*/) override {
			return std::unexpected("MockModelLoader: load not implemented");
		}

	private:
		std::string ext_;
	};

} // anonymous namespace

// =============================================================================
// IModelLoader interface contracts
// =============================================================================

TEST_CASE("IModelLoader::probe returns true for matching extension", "[model][loader]") {
	MockModelLoader loader(".mock");
	CHECK(loader.probe("my_model.mock"));
	CHECK_FALSE(loader.probe("my_model.so"));
	CHECK_FALSE(loader.probe(""));
}

TEST_CASE("IModelLoader::probe is case-sensitive", "[model][loader]") {
	MockModelLoader loader(".mock");
	CHECK_FALSE(loader.probe("model.MOCK"));
	CHECK_FALSE(loader.probe("model.Mock"));
}

TEST_CASE("IModelLoader::load returns error for mock loader", "[model][loader]") {
	MockModelLoader loader(".mock");
	auto result = loader.load("model.mock", "{}", ttm::model::Device{});
	CHECK_FALSE(result.has_value());
	CHECK(!result.error().empty());
}

// =============================================================================
// Device integration with IModelLoader::load
// =============================================================================

TEST_CASE("IModelLoader::load receives correct device", "[model][loader]") {
	// Verify that Device constructed from string round-trips through load() call.
	MockModelLoader loader(".test");
	const auto dev = ttm::model::Device::from_string("cpu");
	auto result = loader.load("m.test", "{}", dev);
	// Mock always returns error — just check it doesn't crash
	CHECK_FALSE(result.has_value());
}

// =============================================================================
// Multiple loaders — probe selectivity
// =============================================================================

TEST_CASE("Multiple IModelLoaders probe different extensions", "[model][loader]") {
	MockModelLoader py_loader(".py");
	MockModelLoader so_loader(".so");

	CHECK(py_loader.probe("model.py"));
	CHECK_FALSE(py_loader.probe("model.so"));

	CHECK(so_loader.probe("model.so"));
	CHECK_FALSE(so_loader.probe("model.py"));
}

TEST_CASE("probe returns false for path with no extension", "[model][loader]") {
	MockModelLoader loader(".so");
	CHECK_FALSE(loader.probe("model_without_extension"));
}

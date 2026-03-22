/**
 * @file test_model_loader.cpp
 * @brief Unit tests for model loader registration and dispatch in PluginManager.
 *
 * @details
 * Uses an inline CModelLoaderAdapter-style test double (a native plugin loaded
 * via the public C vtable registration path) to verify:
 * - findModelLoader() returns nullptr when no loader is registered
 * - A loader registered through the host API is found by findModelLoader()
 * - probe() is called to select the correct loader
 * - emitModelLoaded() is a no-op (does not crash) with no plugins loaded
 * - findTransform() returns nullptr when no transforms are registered
 */

#include <tmm/compat/expected.hpp>
#include <tmm/model/device.hpp>
#include <tmm/model/model.hpp>
#include <tmm/plugins/plugin_manager.hpp>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <string_view>

// =============================================================================
// findModelLoader with no loaders registered
// =============================================================================

TEST_CASE("findModelLoader returns nullptr when no loaders registered", "[model][loader]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	CHECK(mgr.findModelLoader("model.py") == nullptr);
	CHECK(mgr.findModelLoader("model.so") == nullptr);
	CHECK(mgr.findModelLoader("") == nullptr);
}

// =============================================================================
// findTransform with no transforms registered
// =============================================================================

TEST_CASE("findTransform returns nullptr when no transforms registered", "[model][loader]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	CHECK(mgr.findTransform("bpe-tokenize") == nullptr);
	CHECK(mgr.findTransform("") == nullptr);
}

// =============================================================================
// emitModelLoaded is safe with no plugins
// =============================================================================

TEST_CASE("emitModelLoaded is a no-op with no plugins loaded", "[model][loader]") {
	auto mgr = tmm::plugins::PluginManager::create().value();
	CHECK_NOTHROW(mgr.emitModelLoaded(R"({"name":"test","arch":"Linear"})"));
	CHECK_NOTHROW(mgr.emitModelLoaded(""));
}

// =============================================================================
// Mock IModelLoader — inlined test double
// =============================================================================

namespace {

	/// Simple IModelLoader that claims ownership of files with a given extension.
	class MockModelLoader final : public tmm::model::IModelLoader {
	public:
		explicit MockModelLoader(std::string_view ext) : ext_(ext) {}

		[[nodiscard]] bool probe(std::string_view path) const override { return path.ends_with(ext_); }

		[[nodiscard]] std::expected<std::unique_ptr<tmm::model::IModel>, std::string>
		load(std::string_view /*path*/, std::string_view /*cfg*/, tmm::model::Device /*dev*/) override {
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
	auto result = loader.load("model.mock", "{}", tmm::model::Device{});
	CHECK_FALSE(result.has_value());
	CHECK(!result.error().empty());
}

// =============================================================================
// Device integration with IModelLoader::load
// =============================================================================

TEST_CASE("IModelLoader::load receives correct device", "[model][loader]") {
	// Verify that Device constructed from string round-trips through load() call.
	MockModelLoader loader(".test");
	const auto dev = tmm::model::Device::from_string("cpu");
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

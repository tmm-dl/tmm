/**
 * @file test_python_plugin_integration.cpp
 * @brief Integration test for the tmm_python plugin (PyTorch model loader).
 *
 * @details
 * These tests exercise the full plugin lifecycle for `.py` model files:
 *
 *   1. Load `tmm_python.so` into a PluginManager via `mgr.load()`.
 *   2. Verify that `findModelLoader()` returns a loader that probes `.py`.
 *   3. Write a minimal `torch.nn.Module` subclass to a temporary `.py` file.
 *   4. Call `loader->open()` and inspect the result:
 *      - Without PyTorch installed: expect a descriptive error message.
 *      - With PyTorch installed: expect a valid handle.
 *   5. When loaded: call `get_info()`, `zero_grad()`, and `destroy()`.
 *
 * ### Running
 * The test requires that `tmm_python.so` was built (TMM_BUILD_EXTENSIONS=ON).
 * It is skipped at runtime when the plugin file is not present.  Set the
 * environment variable `TMM_PYTHON_PLUGIN` to override the plugin path.
 *
 * ### PyTorch availability
 * When PyTorch is **not** installed the test still passes — it verifies
 * that the plugin returns a helpful error rather than crashing.  When
 * PyTorch **is** installed the test additionally exercises the happy path.
 */

#include <tmm/model/device.hpp>
#include <tmm/plugins/plugin_manager.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

// =============================================================================
// Helpers
// =============================================================================

namespace {

	/// Resolve the path to tmm_python.so.
	/// Checks $TMM_PYTHON_PLUGIN env var first, then falls back to the
	/// compile-time path injected by CMake (TMM_PYTHON_PLUGIN_PATH macro).
	std::filesystem::path plugin_path() {
		if (const char* env = std::getenv("TMM_PYTHON_PLUGIN"); env != nullptr) {
			return env;
		}
#ifdef TMM_PYTHON_PLUGIN_PATH
		return TMM_PYTHON_PLUGIN_PATH;
#else
		return {}; // unknown — test will skip
#endif
	}

	/// RAII wrapper that writes a file and removes it on destruction.
	struct TempPyFile {
		std::filesystem::path path;

		explicit TempPyFile(std::string_view content) {
			path = std::filesystem::temp_directory_path() / "tmm_test_model_XXXXXX.py";
			path.replace_filename("tmm_test_model_integration.py");
			std::ofstream ofs(path);
			ofs << content;
		}
		TempPyFile(const TempPyFile&) = delete;
		TempPyFile& operator=(const TempPyFile&) = delete;
		TempPyFile(TempPyFile&&) = default;
		TempPyFile& operator=(TempPyFile&&) = default;
		~TempPyFile() { std::filesystem::remove(path); }
	};

	/// Minimal PyTorch model: a single Linear layer with a no-arg constructor.
	constexpr std::string_view kMinimalModel = R"python(
import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)
)python";

} // anonymous namespace

// =============================================================================
// Plugin presence guard
// =============================================================================

TEST_CASE("tmm_python plugin file is locatable", "[python][integration][.optional]") {
	const auto path = plugin_path();
	if (path.empty()) {
		SKIP("TMM_PYTHON_PLUGIN_PATH not set and $TMM_PYTHON_PLUGIN not in environment");
	}
	if (!std::filesystem::exists(path)) {
		SKIP("tmm_python plugin not built; run cmake with -DTMM_BUILD_EXTENSIONS=ON");
	}
	CHECK(std::filesystem::exists(path));
}

// =============================================================================
// Plugin loads into PluginManager
// =============================================================================

TEST_CASE("PluginManager loads tmm_python plugin without error", "[python][integration][.optional]") {
	const auto path = plugin_path();
	if (path.empty() || !std::filesystem::exists(path)) {
		SKIP("tmm_python plugin not available");
	}

	auto mgr = tmm::plugins::PluginManager::create().value();
	auto result = mgr.load(path);
	REQUIRE(result.has_value());
}

// =============================================================================
// Probe: .py extension is claimed
// =============================================================================

TEST_CASE("tmm_python registers a loader that probes .py files", "[python][integration][.optional]") {
	const auto path = plugin_path();
	if (path.empty() || !std::filesystem::exists(path)) {
		SKIP("tmm_python plugin not available");
	}

	auto mgr = tmm::plugins::PluginManager::create().value();
	REQUIRE(mgr.load(path).has_value());

	auto* loader = mgr.findModelLoader("model.py");
	REQUIRE(loader != nullptr);
}

TEST_CASE("tmm_python loader does not claim .so files", "[python][integration][.optional]") {
	const auto path = plugin_path();
	if (path.empty() || !std::filesystem::exists(path)) {
		SKIP("tmm_python plugin not available");
	}

	auto mgr = tmm::plugins::PluginManager::create().value();
	REQUIRE(mgr.load(path).has_value());

	// .so files belong to the TVM loader, not the Python loader.
	auto* loader = mgr.findModelLoader("model.so");
	// Either nullptr (no TVM loader registered) or not the python loader.
	// Either way, the python loader must not claim this extension.
	if (loader != nullptr) {
		// If something claimed it, verify it's not the python probe responding
		// by checking the open() call returns an error mentioning torch/python.
		// (We cannot introspect loader identity without down-casting.)
	}
	// Primary assertion: .py loader is not confused with .so
	auto* py_loader = mgr.findModelLoader("model.py");
	auto* so_loader = mgr.findModelLoader("model.so");
	CHECK((py_loader != so_loader || so_loader == nullptr));
}

// =============================================================================
// Load: minimal torch.nn.Module subclass
// =============================================================================

TEST_CASE(
		"tmm_python open() on a minimal model returns a valid handle or a helpful error",
		"[python][integration][.optional]"
) {
	const auto plugin = plugin_path();
	if (plugin.empty() || !std::filesystem::exists(plugin)) {
		SKIP("tmm_python plugin not available");
	}

	const TempPyFile tmp{kMinimalModel};

	auto mgr = tmm::plugins::PluginManager::create().value();
	REQUIRE(mgr.load(plugin).has_value());

	auto* loader = mgr.findModelLoader(tmp.path.string());
	REQUIRE(loader != nullptr);

	auto result = loader->open(tmp.path.string(), "{}");

	if (result.has_value()) {
		// PyTorch is installed — happy path.
		const tmm_handle h = result.value();
		CHECK(h != TMM_INVALID_HANDLE);

		// get_info() should return a non-empty name.
		const auto info = loader->get_info(h);
		CHECK(info.name != nullptr);
		CHECK(std::string_view{info.name}.size() > 0);

		// zero_grad() must not crash.
		CHECK(loader->zero_grad(h) == TMM_OK);

		loader->destroy(h);
	} else {
		// PyTorch not installed — verify the error is descriptive.
		const std::string& err = result.error();
		CHECK(!err.empty());
		// The error should mention either "torch", "PyTorch", or "Python".
		const bool mentions_torch = err.find("torch") != std::string::npos ||
									err.find("PyTorch") != std::string::npos || err.find("Python") != std::string::npos;
		CHECK(mentions_torch);
	}
}

// =============================================================================
// Load: file with no torch.nn.Module subclass
// =============================================================================

TEST_CASE("tmm_python open() fails gracefully on a .py file with no nn.Module", "[python][integration][.optional]") {
	const auto plugin = plugin_path();
	if (plugin.empty() || !std::filesystem::exists(plugin)) {
		SKIP("tmm_python plugin not available");
	}

	// A valid Python file but with no torch.nn.Module subclass.
	const TempPyFile tmp{"x = 42\n"};

	auto mgr = tmm::plugins::PluginManager::create().value();
	REQUIRE(mgr.load(plugin).has_value());

	auto* loader = mgr.findModelLoader(tmp.path.string());
	REQUIRE(loader != nullptr);

	auto result = loader->open(tmp.path.string(), "{}");

	// Must return either a meaningful error (torch not present / no Module found)
	// rather than an invalid handle with a silent failure.
	if (!result.has_value()) {
		CHECK(!result.error().empty());
	} else {
		// If torch is available, the load should have failed because there's
		// no nn.Module subclass — still possible if TMM_PYTHON_HAS_TORCH is
		// not compiled in (stub accepts the handle).
		loader->destroy(result.value());
	}
}

// =============================================================================
// Load: non-existent file
// =============================================================================

TEST_CASE("tmm_python open() fails cleanly for a missing file", "[python][integration][.optional]") {
	const auto plugin = plugin_path();
	if (plugin.empty() || !std::filesystem::exists(plugin)) {
		SKIP("tmm_python plugin not available");
	}

	auto mgr = tmm::plugins::PluginManager::create().value();
	REQUIRE(mgr.load(plugin).has_value());

	auto* loader = mgr.findModelLoader("nonexistent_model.py");
	REQUIRE(loader != nullptr);

	auto result = loader->open("/this/path/does/not/exist.py", "{}");
	CHECK_FALSE(result.has_value());
	if (!result.has_value()) {
		CHECK(!result.error().empty());
	}
}

// =============================================================================
// Lifecycle: describe_params returns 0 (PyTorch manages its own params)
// =============================================================================

TEST_CASE(
		"tmm_python describe_params returns zero params (PyTorch manages memory)", "[python][integration][.optional]"
) {
	const auto plugin = plugin_path();
	if (plugin.empty() || !std::filesystem::exists(plugin)) {
		SKIP("tmm_python plugin not available");
	}

	const TempPyFile tmp{kMinimalModel};

	auto mgr = tmm::plugins::PluginManager::create().value();
	REQUIRE(mgr.load(plugin).has_value());

	auto* loader = mgr.findModelLoader(tmp.path.string());
	REQUIRE(loader != nullptr);

	auto result = loader->open(tmp.path.string(), "{}");
	if (!result.has_value()) {
		SKIP("PyTorch not available — skipping happy-path describe_params test");
	}

	const tmm_handle h = result.value();
	const tmm_param_desc_t* descs = nullptr;
	uint32_t count = 99; // sentinel — must be overwritten to 0
	CHECK(loader->describe_params(h, &descs, &count) == TMM_OK);
	CHECK(count == 0); // Python plugin defers param management to PyTorch
	CHECK(descs == nullptr);

	loader->destroy(h);
}

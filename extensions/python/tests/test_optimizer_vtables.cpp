/**
 * @file test_optimizer_vtables.cpp
 * @brief Integration tests for the AdamW optimizer vtable in ttm_python.
 *
 * @details
 * These tests load `ttm_python.so` via dlopen (or LoadLibrary on Windows),
 * call `ttm_plugin_init` with a mock host API that captures the registered
 * model loader and optimizer vtables, then exercise the AdamW optimizer
 * through the vtable function pointers.
 *
 * ### Running standalone
 * Build with `-DTTM_BUILD_EXTENSIONS=ON` then:
 * @code
 *   ctest --test-dir build --output-on-failure -R "^ttm_python"
 * @endcode
 *
 * ### Running PyTorch-dependent tests
 * The tests tagged `[.optional]` require PyTorch to be installed.  Run them
 * explicitly:
 * @code
 *   ctest --test-dir build -R "^ttm_python" -C ".*optional.*"
 *   # or via Catch2 directly:
 *   ./ttm_python_tests "[python][optimizer]"
 *   ./ttm_python_tests "[python][optimizer][.optional]"
 * @endcode
 *
 * ### Plugin singleton
 * A static `PythonPlugin` instance is shared across all tests in this
 * translation unit.  This avoids repeated `Py_Initialize` / `Py_Finalize`
 * cycles (CPython does not fully support multiple init/finalize cycles within
 * one process).  The plugin is torn down once when the process exits.
 */

#define PY_SSIZE_T_CLEAN // suppress Python.h warning if Python is found
#include <ttm/plugins/abi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

// =============================================================================
// Platform helpers
// =============================================================================

namespace {

void* lib_open(const char* path) {
#ifdef _WIN32
    return static_cast<void*>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* lib_sym(void* lib, const char* name) {
#ifdef _WIN32
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

void lib_close(void* lib) {
    if (lib == nullptr) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(lib));
#else
    dlclose(lib);
#endif
}

// =============================================================================
// Python model file helpers
// =============================================================================

/// Minimal torch.nn.Module subclass — enough for an optimizer to wrap.
static constexpr const char* kMinimalModel = R"python(
import torch

class MinimalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)
)python";

/// RAII helper that writes a temporary .py file and removes it on destruction.
class TempPyFile {
public:
    explicit TempPyFile(const char* content) {
        // Use the object address to ensure a unique filename per instance.
        path_ = std::filesystem::temp_directory_path()
              / ("ttm_opt_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".py");
        std::ofstream f(path_);
        f << content;
    }
    ~TempPyFile() { std::filesystem::remove(path_); }

    TempPyFile(const TempPyFile&)            = delete;
    TempPyFile& operator=(const TempPyFile&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// =============================================================================
// Plugin fixture
// =============================================================================

/// Return the filesystem path to ttm_python.so, from environment or CMake def.
std::filesystem::path python_plugin_path() {
    if (const char* env = std::getenv("TTM_PYTHON_PLUGIN"); env != nullptr) {
        return env;
    }
#ifdef TTM_PYTHON_PLUGIN_PATH
    return TTM_PYTHON_PLUGIN_PATH;
#else
    return {};
#endif
}

/**
 * @brief RAII guard that loads ttm_python.so and collects registered vtables.
 *
 * @details
 * On construction: opens the shared library, calls ttm_plugin_init with a mock
 * host API that records the `register_model_loader` and `register_optimizer`
 * calls.  On destruction: calls ttm_plugin_teardown and closes the library.
 */
class PythonPlugin {
public:
    ttm_model_loader_vtable                        model_loader{};
    std::unordered_map<std::string, ttm_optimizer_vtable> optimizers;
    bool loaded           = false;
    bool has_model_loader = false;

    PythonPlugin() {
        const auto path = python_plugin_path();
        if (path.empty() || !std::filesystem::exists(path)) return;

        lib_ = lib_open(path.string().c_str());
        if (lib_ == nullptr) return;

        // Resolve required entry points
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        fnInit_     = reinterpret_cast<decltype(fnInit_)>(lib_sym(lib_, "ttm_plugin_init"));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        fnTeardown_ = reinterpret_cast<decltype(fnTeardown_)>(lib_sym(lib_, "ttm_plugin_teardown"));

        if (fnInit_ == nullptr) { lib_close(lib_); lib_ = nullptr; return; }

        // Build mock host API — only register_model_loader and register_optimizer
        // are exercised here; all others are noops.
        ttm_host_api api{};
        api.ctx                   = this;
        api.register_source       = &s_noop_register_source;
        api.register_transform    = &s_noop_register_transform;
        api.register_task         = &s_noop_register_task;
        api.register_metric       = &s_noop_register_metric;
        api.register_model_loader = &s_register_model_loader;
        api.notify_model_info     = &s_noop_notify_model_info;
        api.register_scheduler    = &s_noop_register_scheduler;
        api.register_optimizer    = &s_register_optimizer;
        api.log                   = &s_noop_log;
        api.log_metric            = &s_noop_log_metric;
        api.terminal_size         = &s_noop_terminal_size;
        api.alloc                 = &s_alloc;
        api.free                  = &s_free;

        if (fnInit_(&api, "{}", 2) == TTM_OK) {
            loaded = true;
        }
    }

    PythonPlugin(const PythonPlugin&)            = delete;
    PythonPlugin& operator=(const PythonPlugin&) = delete;

    ~PythonPlugin() {
        if (fnTeardown_ != nullptr) fnTeardown_();
        lib_close(lib_);
    }

private:
    void* lib_        = nullptr;
    ttm_error (*fnInit_)(const ttm_host_api*, const char*, uint32_t) = nullptr;
    void      (*fnTeardown_)()                                        = nullptr;

    // --- mock callbacks ---
    static ttm_error s_register_optimizer(void* ctx, const char* name, const ttm_optimizer_vtable* vt) {
        if (ctx == nullptr || name == nullptr || vt == nullptr) return TTM_ERR_ARGS;
        static_cast<PythonPlugin*>(ctx)->optimizers.insert_or_assign(std::string(name), *vt);
        return TTM_OK;
    }
    static ttm_error s_register_model_loader(void* ctx, const ttm_model_loader_vtable* vt) {
        if (ctx == nullptr || vt == nullptr) return TTM_ERR_ARGS;
        auto* self          = static_cast<PythonPlugin*>(ctx);
        self->model_loader  = *vt;
        self->has_model_loader = true;
        return TTM_OK;
    }
    static ttm_error s_noop_register_source(void*, const char**, const ttm_source_vtable*) { return TTM_OK; }
    static ttm_error s_noop_register_transform(void*, const char*, const char**, const ttm_transform_vtable*) { return TTM_OK; }
    static ttm_error s_noop_register_task(void*, const char*, const char**, const ttm_task_vtable*) { return TTM_OK; }
    static ttm_error s_noop_register_metric(void*, const char*, const char**, const ttm_metric_vtable*) { return TTM_OK; }
    static void      s_noop_notify_model_info(void*, const ttm_model_info_t*) {}
    static ttm_error s_noop_register_scheduler(void*, const char*, const ttm_scheduler_vtable*) { return TTM_OK; }
    static void      s_noop_log(void*, ttm_log_level, const char*, uint32_t) {}
    static void      s_noop_log_metric(void*, const char*, uint32_t, float, int32_t) {}
    static void      s_noop_terminal_size(void*, uint32_t* w, uint32_t* h) { if (w) *w = 80; if (h) *h = 24; }
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
    static void* s_alloc(void*, uint32_t n) { return std::malloc(n); }
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
    static void  s_free(void*, void* p) { std::free(p); }
};

/// Singleton — shared across all tests to avoid multiple Py_Initialize/Py_Finalize cycles.
PythonPlugin& python_plugin() {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static PythonPlugin s_plugin;
    return s_plugin;
}

// =============================================================================
// Helpers
// =============================================================================

/// Build a minimal optimizer config JSON.
std::string make_opt_cfg(float lr = 1e-3f, float weight_decay = 0.0f, const char* device = "cpu") {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"lr\":%g,\"weight_decay\":%g,\"device\":\"%s\"}",
        static_cast<double>(lr), static_cast<double>(weight_decay), device);
    return buf;
}

static constexpr float kEps = 1e-5f;

} // anonymous namespace

// =============================================================================
// Tests — no PyTorch required
// =============================================================================

TEST_CASE("Python plugin loads and registers adamw optimizer", "[python][optimizer]") {
    auto& p = python_plugin();
    if (!p.loaded) {
        WARN("ttm_python.so not found — skipping python optimizer tests");
        return;
    }

    SECTION("adamw vtable is registered") {
        CHECK(p.optimizers.count("adamw") == 1);
    }

    SECTION("adamw vtable has non-null function pointers") {
        REQUIRE(p.optimizers.count("adamw") == 1);
        const auto& vt = p.optimizers.at("adamw");
        CHECK(vt.create    != nullptr);
        CHECK(vt.step      != nullptr);
        CHECK(vt.zero_grad != nullptr);
        CHECK(vt.get_lr    != nullptr);
        CHECK(vt.set_lr    != nullptr);
        CHECK(vt.destroy   != nullptr);
    }

    SECTION("model loader vtable is registered") {
        CHECK(p.has_model_loader);
        CHECK(p.model_loader.load    != nullptr);
        CHECK(p.model_loader.destroy != nullptr);
    }
}

TEST_CASE("adamw create with invalid model handle returns error", "[python][optimizer]") {
    auto& p = python_plugin();
    if (!p.loaded) return;
    REQUIRE(p.optimizers.count("adamw") == 1);

    const auto& vt  = p.optimizers.at("adamw");
    const auto  cfg = make_opt_cfg();
    char err[256]{};

    const ttm_handle h = vt.create(
        TTM_INVALID_HANDLE,
        nullptr, 0, nullptr,
        cfg.c_str(), static_cast<uint32_t>(cfg.size()),
        err, sizeof(err));

    CHECK(h == TTM_INVALID_HANDLE);
    CHECK(err[0] != '\0'); // error message must be set
}

TEST_CASE("adamw step/zero_grad/get_lr on invalid handle return sentinel values", "[python][optimizer]") {
    auto& p = python_plugin();
    if (!p.loaded) return;
    REQUIRE(p.optimizers.count("adamw") == 1);

    const auto& vt = p.optimizers.at("adamw");
    CHECK(vt.step(TTM_INVALID_HANDLE)      == TTM_ERR_NOT_FOUND);
    CHECK(vt.zero_grad(TTM_INVALID_HANDLE) == TTM_ERR_NOT_FOUND);
    CHECK(vt.get_lr(TTM_INVALID_HANDLE)    == 0.0f);
}

TEST_CASE("adamw destroy on invalid handle is safe", "[python][optimizer]") {
    auto& p = python_plugin();
    if (!p.loaded) return;
    REQUIRE(p.optimizers.count("adamw") == 1);

    const auto& vt = p.optimizers.at("adamw");
    // Must not crash
    vt.destroy(TTM_INVALID_HANDLE);
    vt.destroy(static_cast<ttm_handle>(-2));
}

// =============================================================================
// Tests — require PyTorch at runtime (excluded from default ctest run)
// =============================================================================

TEST_CASE("adamw full lifecycle with a PyTorch model", "[python][optimizer][.optional]") {
    auto& p = python_plugin();
    if (!p.loaded) return;
    REQUIRE(p.has_model_loader);
    REQUIRE(p.optimizers.count("adamw") == 1);

    TempPyFile tmp(kMinimalModel);
    const std::string path = tmp.path().string();

    char err[512]{};
    const ttm_handle model_h = p.model_loader.load(
        path.c_str(), static_cast<uint32_t>(path.size()),
        "{}", 2, err, sizeof(err));

    if (model_h == TTM_INVALID_HANDLE) {
        WARN("PyTorch not available (" << err << ") — skipping lifecycle test");
        return;
    }

    const auto& vt  = p.optimizers.at("adamw");
    const auto  cfg = make_opt_cfg(1e-3f);

    const ttm_handle opt_h = vt.create(
        model_h, nullptr, 0, nullptr,
        cfg.c_str(), static_cast<uint32_t>(cfg.size()),
        err, sizeof(err));
    REQUIRE(opt_h != TTM_INVALID_HANDLE);

    // Initial LR must match what was configured
    CHECK_THAT(vt.get_lr(opt_h), Catch::Matchers::WithinAbs(1e-3f, kEps));

    // set_lr / get_lr round-trip
    vt.set_lr(opt_h, 5e-4f);
    CHECK_THAT(vt.get_lr(opt_h), Catch::Matchers::WithinAbs(5e-4f, kEps));

    // zero_grad and step must not error (no real forward pass — gradients are
    // zero, but the optimizer step is still a valid no-op)
    CHECK(vt.zero_grad(opt_h) == TTM_OK);
    CHECK(vt.step(opt_h)      == TTM_OK);

    vt.destroy(opt_h);
    p.model_loader.destroy(model_h);
}

TEST_CASE("adamw multiple optimizer handles are independent", "[python][optimizer][.optional]") {
    auto& p = python_plugin();
    if (!p.loaded) return;
    REQUIRE(p.has_model_loader);
    REQUIRE(p.optimizers.count("adamw") == 1);

    TempPyFile tmp(kMinimalModel);
    const std::string path = tmp.path().string();

    char err[512]{};
    const ttm_handle m1 = p.model_loader.load(
        path.c_str(), static_cast<uint32_t>(path.size()), "{}", 2, err, sizeof(err));
    if (m1 == TTM_INVALID_HANDLE) {
        WARN("PyTorch not available — skipping independence test");
        return;
    }
    const ttm_handle m2 = p.model_loader.load(
        path.c_str(), static_cast<uint32_t>(path.size()), "{}", 2, err, sizeof(err));
    REQUIRE(m2 != TTM_INVALID_HANDLE);

    const auto& vt   = p.optimizers.at("adamw");
    const auto  cfg1 = make_opt_cfg(1e-3f);
    const auto  cfg2 = make_opt_cfg(1e-2f);

    const ttm_handle opt1 = vt.create(m1, nullptr, 0, nullptr,
        cfg1.c_str(), static_cast<uint32_t>(cfg1.size()), err, sizeof(err));
    const ttm_handle opt2 = vt.create(m2, nullptr, 0, nullptr,
        cfg2.c_str(), static_cast<uint32_t>(cfg2.size()), err, sizeof(err));

    REQUIRE(opt1 != TTM_INVALID_HANDLE);
    REQUIRE(opt2 != TTM_INVALID_HANDLE);
    REQUIRE(opt1 != opt2);

    // Each handle reports its own configured LR
    CHECK_THAT(vt.get_lr(opt1), Catch::Matchers::WithinAbs(1e-3f, kEps));
    CHECK_THAT(vt.get_lr(opt2), Catch::Matchers::WithinAbs(1e-2f, kEps));

    // Updating one does not affect the other
    vt.set_lr(opt1, 9e-4f);
    CHECK_THAT(vt.get_lr(opt1), Catch::Matchers::WithinAbs(9e-4f, kEps));
    CHECK_THAT(vt.get_lr(opt2), Catch::Matchers::WithinAbs(1e-2f, kEps));

    vt.destroy(opt1);
    vt.destroy(opt2);
    p.model_loader.destroy(m1);
    p.model_loader.destroy(m2);
}

TEST_CASE("adamw lr integrates with scheduler-driven updates", "[python][optimizer][.optional]") {
    auto& p = python_plugin();
    if (!p.loaded) return;
    REQUIRE(p.has_model_loader);
    REQUIRE(p.optimizers.count("adamw") == 1);

    TempPyFile tmp(kMinimalModel);
    const std::string path = tmp.path().string();

    char err[512]{};
    const ttm_handle model_h = p.model_loader.load(
        path.c_str(), static_cast<uint32_t>(path.size()), "{}", 2, err, sizeof(err));
    if (model_h == TTM_INVALID_HANDLE) {
        WARN("PyTorch not available — skipping scheduler integration test");
        return;
    }

    const auto& vt  = p.optimizers.at("adamw");
    const auto  cfg = make_opt_cfg(1.0f); // start at lr=1.0 for easy arithmetic

    const ttm_handle opt_h = vt.create(
        model_h, nullptr, 0, nullptr,
        cfg.c_str(), static_cast<uint32_t>(cfg.size()),
        err, sizeof(err));
    REQUIRE(opt_h != TTM_INVALID_HANDLE);

    // Simulate a scheduler halving the LR at each step
    for (float expected = 1.0f; expected > 1e-3f; expected *= 0.5f) {
        CHECK_THAT(vt.get_lr(opt_h), Catch::Matchers::WithinAbs(expected, kEps));
        vt.set_lr(opt_h, expected * 0.5f);
    }

    vt.destroy(opt_h);
    p.model_loader.destroy(model_h);
}

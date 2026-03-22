/**
 * @file test_scheduler_vtables.cpp
 * @brief Integration tests for the built-in LR scheduler vtables in ttm_core.
 *
 * @details
 * These tests load `ttm_core.so` via dlopen (or LoadLibrary on Windows),
 * call `ttm_plugin_init` with a mock host API that captures all registered
 * scheduler vtables, and then exercise the scheduler math through the vtable
 * function pointers.
 *
 * ### Running standalone
 * Build with `-DTTM_BUILD_EXTENSIONS=ON` then:
 * @code
 *   ctest --test-dir build --output-on-failure -R "^ttm_core"
 * @endcode
 *
 * ### Scheduler coverage
 * - constant      — LR never changes
 * - step          — multiplicative decay every `step_size` steps
 * - linear        — linear decay from `base_lr` to `min_lr`
 * - cosine        — cosine annealing to `min_lr`
 * - cosine_warmup — linear warmup then cosine annealing
 */

#define PY_SSIZE_T_CLEAN // suppress Python.h warning if Python is found
#include <ttm/plugins/abi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
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
		if (lib == nullptr)
			return;
#ifdef _WIN32
		FreeLibrary(static_cast<HMODULE>(lib));
#else
		dlclose(lib);
#endif
	}

	// =============================================================================
	// Plugin fixture
	// =============================================================================

	/// Return the path to ttm_core.so, from environment or CMake compile-def.
	std::filesystem::path core_plugin_path() {
		if (const char* env = std::getenv("TTM_CORE_PLUGIN"); env != nullptr) {
			return env;
		}
#ifdef TTM_CORE_PLUGIN_PATH
		return TTM_CORE_PLUGIN_PATH;
#else
		return {};
#endif
	}

	/**
	 * @brief RAII guard that loads ttm_core.so and collects registered schedulers.
	 *
	 * @details
	 * On construction: opens the shared library, calls ttm_plugin_init with a mock
	 * host API that records each `register_scheduler` call.  On destruction: calls
	 * ttm_plugin_teardown and closes the library.
	 */
	class CorePlugin {
	public:
		std::unordered_map<std::string, ttm_scheduler_vtable> schedulers;
		bool loaded = false;

		CorePlugin() {
			const auto path = core_plugin_path();
			if (path.empty() || !std::filesystem::exists(path))
				return;

			lib_ = lib_open(path.string().c_str());
			if (lib_ == nullptr)
				return;

			// Resolve required entry points
			// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
			fnInit_ = reinterpret_cast<decltype(fnInit_)>(lib_sym(lib_, "ttm_plugin_init"));
			// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
			fnTeardown_ = reinterpret_cast<decltype(fnTeardown_)>(lib_sym(lib_, "ttm_plugin_teardown"));

			if (fnInit_ == nullptr) {
				lib_close(lib_);
				lib_ = nullptr;
				return;
			}

			// Build mock host API — only register_scheduler is exercised here
			ttm_host_api api{};
			api.ctx = this;
			api.register_source = &s_noop_register_source;
			api.register_transform = &s_noop_register_transform;
			api.register_task = &s_noop_register_task;
			api.register_metric = &s_noop_register_metric;
			api.register_model_loader = &s_noop_register_model_loader;
			api.notify_model_info = &s_noop_notify_model_info;
			api.register_scheduler = &s_register_scheduler;
			api.log = &s_noop_log;
			api.log_metric = &s_noop_log_metric;
			api.terminal_size = &s_noop_terminal_size;
			api.alloc = &s_alloc;
			api.free = &s_free;

			if (fnInit_(&api, "{}", 2) == TTM_OK) {
				loaded = true;
			}
		}

		CorePlugin(const CorePlugin&) = delete;
		CorePlugin& operator=(const CorePlugin&) = delete;

		~CorePlugin() {
			if (fnTeardown_ != nullptr)
				fnTeardown_();
			lib_close(lib_);
		}

	private:
		void* lib_ = nullptr;
		ttm_error (*fnInit_)(const ttm_host_api*, const char*, uint32_t) = nullptr;
		void (*fnTeardown_)() = nullptr;

		// --- mock callbacks ---
		static ttm_error s_register_scheduler(void* ctx, const char* name, const ttm_scheduler_vtable* vt) {
			if (ctx == nullptr || name == nullptr || vt == nullptr)
				return TTM_ERR_ARGS;
			static_cast<CorePlugin*>(ctx)->schedulers.insert_or_assign(std::string(name), *vt);
			return TTM_OK;
		}
		static ttm_error s_noop_register_source(void*, const char**, const ttm_source_vtable*) { return TTM_OK; }
		static ttm_error s_noop_register_transform(void*, const char*, const char**, const ttm_transform_vtable*) {
			return TTM_OK;
		}
		static ttm_error s_noop_register_task(void*, const char*, const char**, const ttm_task_vtable*) {
			return TTM_OK;
		}
		static ttm_error s_noop_register_metric(void*, const char*, const char**, const ttm_metric_vtable*) {
			return TTM_OK;
		}
		static ttm_error s_noop_register_model_loader(void*, const ttm_model_loader_vtable*) { return TTM_OK; }
		static void s_noop_notify_model_info(void*, const ttm_model_info_t*) {}
		static void s_noop_log(void*, ttm_log_level, const char*, uint32_t) {}
		static void s_noop_log_metric(void*, const char*, uint32_t, float, int32_t) {}
		static void s_noop_terminal_size(void*, uint32_t* w, uint32_t* h) {
			if (w)
				*w = 80;
			if (h)
				*h = 24;
		}
		// NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
		static void* s_alloc(void*, uint32_t n) { return std::malloc(n); }
		// NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory)
		static void s_free(void*, void* p) { std::free(p); }
	};

	// =============================================================================
	// Helpers
	// =============================================================================

	/// Build a scheduler config JSON string from the given fields.
	std::string make_cfg(
			int64_t warmup = 0, float min_lr = 0.0f, int64_t step_size = 1, float gamma = 0.1f, int64_t total_steps = 0
	) {
		char buf[256];
		std::snprintf(
				buf, sizeof(buf),
				"{\"warmup_steps\":%lld,\"min_lr\":%f,\"step_size\":%lld,\"gamma\":%f,\"total_steps\":%lld}",
				static_cast<long long>(warmup), static_cast<double>(min_lr), static_cast<long long>(step_size),
				static_cast<double>(gamma), static_cast<long long>(total_steps)
		);
		return buf;
	}

	static constexpr float kEps = 1e-5f;

} // anonymous namespace

// =============================================================================
// Tests
// =============================================================================

TEST_CASE("ttm_core plugin loads and registers schedulers", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded) {
		WARN("ttm_core.so not found — skipping core plugin scheduler tests");
		return;
	}

	SECTION("all five built-in schedulers are registered") {
		CHECK(p.schedulers.count("constant") == 1);
		CHECK(p.schedulers.count("step") == 1);
		CHECK(p.schedulers.count("linear") == 1);
		CHECK(p.schedulers.count("cosine") == 1);
		CHECK(p.schedulers.count("cosine_warmup") == 1);
	}

	SECTION("each vtable has non-null create / step / destroy") {
		for (const auto& [name, vt] : p.schedulers) {
			INFO("scheduler: " << name);
			CHECK(vt.create != nullptr);
			CHECK(vt.step != nullptr);
			CHECK(vt.destroy != nullptr);
		}
	}
}

TEST_CASE("constant scheduler returns base_lr at every step", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("constant");
	const float base = 1e-3f;
	const auto h = vt.create(base, "{}", 2);
	REQUIRE(h != TTM_INVALID_HANDLE);

	CHECK_THAT(vt.step(h, 0), Catch::Matchers::WithinAbs(base, kEps));
	CHECK_THAT(vt.step(h, 100), Catch::Matchers::WithinAbs(base, kEps));
	CHECK_THAT(vt.step(h, 999), Catch::Matchers::WithinAbs(base, kEps));

	vt.destroy(h);
}

TEST_CASE("step scheduler decays by gamma every step_size steps", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("step");
	const float base = 1.0f;
	const float gamma = 0.5f;
	const auto cfg = make_cfg(0, 0.0f, 3, gamma, 0);
	const auto h = vt.create(base, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	REQUIRE(h != TTM_INVALID_HANDLE);

	// Steps 0–2: no decay yet
	CHECK_THAT(vt.step(h, 0), Catch::Matchers::WithinAbs(1.0f, kEps));
	CHECK_THAT(vt.step(h, 2), Catch::Matchers::WithinAbs(1.0f, kEps));
	// Step 3: one decay → 1.0 * 0.5 = 0.5
	CHECK_THAT(vt.step(h, 3), Catch::Matchers::WithinAbs(0.5f, kEps));
	// Step 6: two decays → 0.25
	CHECK_THAT(vt.step(h, 6), Catch::Matchers::WithinAbs(0.25f, kEps));

	vt.destroy(h);
}

TEST_CASE("step scheduler respects min_lr floor", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("step");
	const auto cfg = make_cfg(0, 0.1f, 1, 0.1f, 0); // decay every step, floor=0.1
	const auto h = vt.create(1.0f, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	REQUIRE(h != TTM_INVALID_HANDLE);

	// After many steps the LR should not go below min_lr=0.1
	const float lr_late = vt.step(h, 100);
	CHECK(lr_late >= 0.1f - kEps);

	vt.destroy(h);
}

TEST_CASE("linear scheduler: full decay from base_lr to min_lr", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("linear");
	const float base = 1.0f;
	const float minl = 0.0f;
	const auto cfg = make_cfg(0, minl, 1, 0.1f, 100); // total_steps=100
	const auto h = vt.create(base, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	REQUIRE(h != TTM_INVALID_HANDLE);

	// Step 0: full base LR
	CHECK_THAT(vt.step(h, 0), Catch::Matchers::WithinAbs(1.0f, kEps));
	// Step 50: halfway → 0.5
	CHECK_THAT(vt.step(h, 50), Catch::Matchers::WithinAbs(0.5f, kEps));
	// Step 100: fully decayed → 0.0 (= min_lr)
	CHECK_THAT(vt.step(h, 100), Catch::Matchers::WithinAbs(0.0f, kEps));

	vt.destroy(h);
}

TEST_CASE("linear scheduler with warmup", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("linear");
	const float base = 1.0f;
	const auto cfg = make_cfg(10, 0.0f, 1, 0.1f, 110); // warmup=10, total=110
	const auto h = vt.create(base, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	REQUIRE(h != TTM_INVALID_HANDLE);

	// During warmup: linear ramp 0→base_lr
	CHECK_THAT(vt.step(h, 0), Catch::Matchers::WithinAbs(0.0f, kEps));
	CHECK_THAT(vt.step(h, 5), Catch::Matchers::WithinAbs(0.5f, kEps));	// half warmup
	CHECK_THAT(vt.step(h, 10), Catch::Matchers::WithinAbs(1.0f, kEps)); // end of warmup

	vt.destroy(h);
}

TEST_CASE("cosine scheduler: base_lr at step 0, min_lr at total_steps", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("cosine");
	const float base = 1.0f;
	const float minl = 0.0f;
	const auto cfg = make_cfg(0, minl, 1, 0.1f, 100);
	const auto h = vt.create(base, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	REQUIRE(h != TTM_INVALID_HANDLE);

	// At step 0: LR = base_lr
	CHECK_THAT(vt.step(h, 0), Catch::Matchers::WithinAbs(base, kEps));
	// At step total_steps: LR = min_lr
	CHECK_THAT(vt.step(h, 100), Catch::Matchers::WithinAbs(minl, kEps));
	// At step total_steps/2: LR ≈ 0.5*(base+min) = 0.5
	CHECK_THAT(vt.step(h, 50), Catch::Matchers::WithinAbs(0.5f, kEps));

	vt.destroy(h);
}

TEST_CASE("cosine scheduler with non-zero min_lr", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("cosine");
	const float base = 1.0f;
	const float minl = 0.2f;
	const auto cfg = make_cfg(0, minl, 1, 0.1f, 100);
	const auto h = vt.create(base, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	REQUIRE(h != TTM_INVALID_HANDLE);

	CHECK_THAT(vt.step(h, 0), Catch::Matchers::WithinAbs(base, kEps));
	CHECK_THAT(vt.step(h, 100), Catch::Matchers::WithinAbs(minl, kEps));
	// Midpoint: min_lr + 0.5*(base-min_lr)*(1+cos(π*0.5)) = 0.2 + 0.5*0.8*1 = 0.6
	CHECK_THAT(vt.step(h, 50), Catch::Matchers::WithinAbs(0.6f, kEps));

	vt.destroy(h);
}

TEST_CASE("cosine with total_steps=0 returns base_lr", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("cosine");
	const auto h = vt.create(0.5f, "{}", 2);
	REQUIRE(h != TTM_INVALID_HANDLE);

	CHECK_THAT(vt.step(h, 0), Catch::Matchers::WithinAbs(0.5f, kEps));
	CHECK_THAT(vt.step(h, 999), Catch::Matchers::WithinAbs(0.5f, kEps));

	vt.destroy(h);
}

TEST_CASE("cosine_warmup: ramp during warmup then cosine decay", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("cosine_warmup");
	const float base = 1.0f;
	const float minl = 0.0f;
	// warmup_steps=20, total_steps=120 → 100 decay steps
	const auto cfg = make_cfg(20, minl, 1, 0.1f, 120);
	const auto h = vt.create(base, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	REQUIRE(h != TTM_INVALID_HANDLE);

	// Warmup: step 0 → LR=0
	CHECK_THAT(vt.step(h, 0), Catch::Matchers::WithinAbs(0.0f, kEps));
	// Warmup midpoint: step 10 → LR=0.5
	CHECK_THAT(vt.step(h, 10), Catch::Matchers::WithinAbs(0.5f, kEps));
	// End of warmup: step 20 → LR=base_lr=1.0
	CHECK_THAT(vt.step(h, 20), Catch::Matchers::WithinAbs(1.0f, kEps));
	// Cosine midpoint: step 70 (20 warmup + 50/100 of decay) → LR≈0.5
	CHECK_THAT(vt.step(h, 70), Catch::Matchers::WithinAbs(0.5f, kEps));
	// End of decay: step 120 → LR=min_lr=0.0
	CHECK_THAT(vt.step(h, 120), Catch::Matchers::WithinAbs(0.0f, kEps));

	vt.destroy(h);
}

TEST_CASE("cosine_warmup with zero warmup acts like plain cosine", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt_cw = p.schedulers.at("cosine_warmup");
	const auto& vt_c = p.schedulers.at("cosine");
	const float base = 0.8f;
	const float minl = 0.1f;
	const auto cfg = make_cfg(0, minl, 1, 0.1f, 200);

	const auto h_cw = vt_cw.create(base, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	const auto h_c = vt_c.create(base, cfg.c_str(), static_cast<uint32_t>(cfg.size()));
	REQUIRE(h_cw != TTM_INVALID_HANDLE);
	REQUIRE(h_c != TTM_INVALID_HANDLE);

	for (int64_t step : {0, 1, 50, 100, 150, 200}) {
		INFO("step=" << step);
		CHECK_THAT(vt_cw.step(h_cw, step), Catch::Matchers::WithinAbs(vt_c.step(h_c, step), kEps));
	}

	vt_cw.destroy(h_cw);
	vt_c.destroy(h_c);
}

TEST_CASE("multiple scheduler handles are independent", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	const auto& vt = p.schedulers.at("constant");
	const auto h1 = vt.create(0.1f, "{}", 2);
	const auto h2 = vt.create(0.9f, "{}", 2);
	REQUIRE(h1 != TTM_INVALID_HANDLE);
	REQUIRE(h2 != TTM_INVALID_HANDLE);
	REQUIRE(h1 != h2);

	CHECK_THAT(vt.step(h1, 0), Catch::Matchers::WithinAbs(0.1f, kEps));
	CHECK_THAT(vt.step(h2, 0), Catch::Matchers::WithinAbs(0.9f, kEps));

	vt.destroy(h1);
	vt.destroy(h2);
}

TEST_CASE("destroy on invalid handle is safe", "[core][scheduler]") {
	CorePlugin p;
	if (!p.loaded)
		return;

	// Should not crash
	const auto& vt = p.schedulers.at("constant");
	vt.destroy(TTM_INVALID_HANDLE);
	vt.destroy(static_cast<ttm_handle>(-2));
}

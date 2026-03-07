/**
 * @file wasm_loader.cpp
 * @brief WAMR-based plugin loader implementation.
 *
 * @details
 * All WAMR API usage is confined to this translation unit so that the rest
 * of the plugin library does not pull in WAMR headers.
 */

#include "wasm_loader.hpp"

#include <ttm/plugins/abi.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <expected>
#include <fstream>
#include <string>
#include <vector>

#include <wasm_export.h>

namespace ttm::plugins {

	/* =========================================================================
 * Runtime init / destroy — refcounted so multiple PluginManager instances
 * (e.g. in unit tests) are safe.
 * ====================================================================== */

	namespace {
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) -- module-level refcount is intentionally mutable; no alternative without dynamic allocation or thread_local
		std::atomic<int> g_wamr_refcount{0};
	} // namespace

	std::expected<void, std::string> wasm_loader_init() {
		if (g_wamr_refcount.fetch_add(1, std::memory_order_acq_rel) == 0) {
			RuntimeInitArgs args{};
			args.mem_alloc_type = Alloc_With_System_Allocator;

			if (!wasm_runtime_full_init(&args)) { // NOLINT(readability-implicit-bool-conversion) -- WAMR API returns bool-like int
				g_wamr_refcount.fetch_sub(1, std::memory_order_acq_rel);
				return std::unexpected("wasm_loader_init: wasm_runtime_full_init failed");
			}
		}
		return {};
	}

	void wasm_loader_destroy() {
		if (g_wamr_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			wasm_runtime_destroy();
		}
	}

	/* =========================================================================
 * String helpers
 * ====================================================================== */

	std::expected<void, std::string>
	// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- out_wasm_ptr and out_len are output parameters with distinct semantics (WASM address vs byte count); names make this clear
	wasm_push_string(wasm_module_inst_t inst, std::string_view str, uint32_t& out_wasm_ptr, uint32_t& out_len) {
		const auto len = static_cast<uint32_t>(str.size());
		void* native_ptr = nullptr;
		const uint32_t wasm_ptr = wasm_runtime_module_malloc(inst, len, &native_ptr);
		if (wasm_ptr == 0) {
			return std::unexpected(
					"wasm_push_string: failed to allocate " + std::to_string(len) + " bytes in WASM linear memory"
			);
		}
		std::memcpy(native_ptr, str.data(), len);
		out_wasm_ptr = wasm_ptr;
		out_len = len;
		return {};
	}

	/* =========================================================================
 * Host import registrations
 *
 * These are the functions exported by the host under the "ttm" module
 * namespace.  They are registered before the WASM module is instantiated so
 * that the linker can resolve the plugin's imports.
 *
 * Signature strings use WAMR's compact notation:
 *   i = i32, I = i64, f = f32, F = f64, * = pointer (i32 in wasm32)
 * ====================================================================== */

	namespace {

		/* ---------- log --------------------------------------------------------- */
		// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- signature is fixed by WAMR NativeSymbol ABI; parameter names clearly distinguish them
		void host_log(wasm_exec_env_t env, uint32_t level, uint32_t msg_ptr, uint32_t msg_len) {
			auto* inst = wasm_runtime_get_module_inst(env);
			const auto* msg = static_cast<const char*>(wasm_runtime_addr_app_to_native(inst, msg_ptr));
			if (msg == nullptr) {
				return;
			}

			const auto* api = static_cast<const ttm_host_api*>(wasm_runtime_get_user_data(env));
			if (api != nullptr && api->log != nullptr) {
				api->log(api->ctx, static_cast<ttm_log_level>(level), msg, msg_len);
			}
		}

		/* ---------- alloc ------------------------------------------------------- */
		uint32_t host_alloc(wasm_exec_env_t env, uint32_t size) {
			auto* inst = wasm_runtime_get_module_inst(env);
			void* native_ptr = nullptr;
			return wasm_runtime_module_malloc(inst, size, &native_ptr);
		}

		/* ---------- free -------------------------------------------------------- */
		void host_free(wasm_exec_env_t env, uint32_t wasm_ptr) {
			auto* inst = wasm_runtime_get_module_inst(env);
			wasm_runtime_module_free(inst, wasm_ptr);
		}

		/* ---------- register_source -------------------------------------------- */
		int32_t
		host_register_source(wasm_exec_env_t /*env*/, uint32_t /*schemes_ptr*/, uint32_t /*vtable_ptr*/) {
			/* TODO: unmarshal the scheme list and vtable from WASM linear memory,
     * construct a CSourceAdapter, and forward to PluginManager via the
     * ttm_host_api ctx.  Stubbed for the initial build. */
			return TTM_ERR_UNSUPPORTED;
		}

		/* ---------- NativeSymbol table ----------------------------------------- */
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables) -- WAMR requires a mutable NativeSymbol[] passed to wasm_runtime_register_natives
		NativeSymbol ttm_native_symbols[] = {
				/* { "export_name", func_ptr, "signature", attachment } */
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- WAMR NativeSymbol API requires void* function pointers; no safer alternative
			{"ttm_log", reinterpret_cast<void*>(host_log), "(iii)", nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
			{"ttm_alloc", reinterpret_cast<void*>(host_alloc), "(i)i", nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
			{"ttm_free", reinterpret_cast<void*>(host_free), "(i)", nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
			{"ttm_register_source", reinterpret_cast<void*>(host_register_source), "(ii)i", nullptr},
		};

	} // anonymous namespace

	/* =========================================================================
 * Load / unload
 * ====================================================================== */

	std::expected<void, std::string> wasm_loader_load(
			const std::filesystem::path& path, std::string_view config_json, const ttm_host_api& host_api,
			Plugin& plugin
	) {
		/* 1. Read file bytes ------------------------------------------------- */
		std::ifstream file(path, std::ios::binary);
		if (!file) {
			return std::unexpected("wasm_loader_load: cannot open '" + path.string() + "'");
		}
		const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

		/* 2. Register host symbols ------------------------------------------- */
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index) -- WAMR API requires array-to-pointer decay for NativeSymbol table
		if (!wasm_runtime_register_natives( // NOLINT(readability-implicit-bool-conversion) -- WAMR API returns bool-like int
					"ttm", ttm_native_symbols, sizeof(ttm_native_symbols) / sizeof(ttm_native_symbols[0]) // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
			)) {
			return std::unexpected("wasm_loader_load: failed to register host symbols");
		}

		/* 3. Load (compile) module ------------------------------------------- */
		constexpr uint32_t kErrBufSize = 256;
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers) -- WAMR API requires a char[] error buffer of fixed size
	char error_buf[kErrBufSize] = {};
		plugin.module = wasm_runtime_load(
				const_cast<uint8_t*>(bytes.data()), // NOLINT(cppcoreguidelines-pro-type-const-cast) -- WAMR does not modify bytes
				static_cast<uint32_t>(bytes.size()),
				error_buf, // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
				kErrBufSize
		);
		if (plugin.module == nullptr) {
			return std::unexpected(std::string("wasm_loader_load: wasm_runtime_load failed: ") + error_buf); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
		}

		/* 4. Instantiate ----------------------------------------------------- */
		constexpr uint32_t STACK_SIZE = 512 * 1024;		/* 512 KB */
		constexpr uint32_t HEAP_SIZE = 4 * 1024 * 1024; /* 4 MB */
		plugin.inst = wasm_runtime_instantiate(
			plugin.module, STACK_SIZE, HEAP_SIZE,
			error_buf, // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
			kErrBufSize
	);
		if (plugin.inst == nullptr) {
			wasm_runtime_unload(plugin.module);
			plugin.module = nullptr;
			return std::unexpected(std::string("wasm_loader_load: instantiate failed: ") + error_buf); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
		}

		/* 5. Create execution environment ------------------------------------ */
		plugin.env = wasm_runtime_create_exec_env(plugin.inst, STACK_SIZE);
		if (plugin.env == nullptr) {
			wasm_runtime_deinstantiate(plugin.inst);
			wasm_runtime_unload(plugin.module);
			plugin.inst = nullptr;
			plugin.module = nullptr;
			return std::unexpected("wasm_loader_load: failed to create exec env");
		}

		/* Attach host API pointer so host callbacks can retrieve it */
		wasm_runtime_set_user_data(plugin.env, const_cast<ttm_host_api*>(&host_api)); // NOLINT(cppcoreguidelines-pro-type-const-cast) -- WAMR user_data is void*; WAMR does not modify it

		/* 6. Verify ABI version ---------------------------------------------- */
		auto* fn_info = wasm_runtime_lookup_function(plugin.inst, "ttm_plugin_get_info");
		if (fn_info == nullptr) {
			wasm_loader_unload(plugin);
			return std::unexpected("wasm_loader_load: '" + path.string() + "' does not export ttm_plugin_get_info");
		}
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- WAMR call ABI requires C array
	uint32_t info_args[1] = {};
		if (!wasm_runtime_call_wasm(plugin.env, fn_info, 0, info_args)) { // NOLINT(readability-implicit-bool-conversion,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
			wasm_loader_unload(plugin);
			return std::unexpected(
					"wasm_loader_load: ttm_plugin_get_info call failed: " +
					std::string(wasm_runtime_get_exception(plugin.inst))
			);
		}
		const auto* info =
				static_cast<const ttm_plugin_info*>(wasm_runtime_addr_app_to_native(plugin.inst, info_args[0]));
		if (info == nullptr) {
			wasm_loader_unload(plugin);
			return std::unexpected("wasm_loader_load: ttm_plugin_get_info returned null");
		}
		if (info->abiVersion != TTM_ABI_VERSION) {
			wasm_loader_unload(plugin);
			return std::unexpected(
					"wasm_loader_load: plugin ABI version mismatch (plugin=" + std::to_string(info->abiVersion) +
					", host=" + std::to_string(TTM_ABI_VERSION) + ")"
			);
		}

		/* 7. Call ttm_plugin_init -------------------------------------------- */
		auto* fn_init = wasm_runtime_lookup_function(plugin.inst, "ttm_plugin_init");
		if (fn_init == nullptr) {
			wasm_loader_unload(plugin);
			return std::unexpected("wasm_loader_load: '" + path.string() + "' does not export ttm_plugin_init");
		}

		/* Push config JSON into WASM linear memory */
		uint32_t config_wasm_ptr = 0;
		uint32_t config_len = 0;
		if (auto r = wasm_push_string(plugin.inst, config_json, config_wasm_ptr, config_len); !r) {
			wasm_loader_unload(plugin);
			return std::unexpected(r.error());
		}

		/* Push host_api struct into WASM linear memory */
		void* api_native_ptr = nullptr;
		const uint32_t api_wasm_ptr = wasm_runtime_module_malloc(plugin.inst, sizeof(ttm_host_api), &api_native_ptr);
		if (api_wasm_ptr == 0) {
			wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
			wasm_loader_unload(plugin);
			return std::unexpected("wasm_loader_load: failed to allocate host_api in WASM heap");
		}
		std::memcpy(api_native_ptr, &host_api, sizeof(ttm_host_api));

		/* init(host_api_ptr, config_ptr, config_len) → i32 */
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- WAMR call ABI requires C array
	uint32_t init_args[3] = {api_wasm_ptr, config_wasm_ptr, config_len};
		if (!wasm_runtime_call_wasm(plugin.env, fn_init, 3, init_args)) { // NOLINT(readability-implicit-bool-conversion,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
			wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
			wasm_runtime_module_free(plugin.inst, api_wasm_ptr);
			wasm_loader_unload(plugin);
			return std::unexpected(
					"wasm_loader_load: ttm_plugin_init failed: " + std::string(wasm_runtime_get_exception(plugin.inst))
			);
		}
		const auto init_result = static_cast<ttm_error>(init_args[0]);
		if (init_result != TTM_OK) {
			wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
			wasm_runtime_module_free(plugin.inst, api_wasm_ptr);
			wasm_loader_unload(plugin);
			return std::unexpected(
					"wasm_loader_load: ttm_plugin_init returned error " + std::to_string(static_cast<int>(init_result))
			);
		}

		/* Free temporaries in WASM heap */
		wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
		wasm_runtime_module_free(plugin.inst, api_wasm_ptr);

		/* 8. Resolve optional lifecycle hooks -------------------------------- */
		auto lookup = [&](const char* name) -> wasm_function_inst_t {
			return wasm_runtime_lookup_function(plugin.inst, name);
		};
		plugin.fnFitBegin = lookup("ttm_on_fit_begin");
		plugin.fnEpochBegin = lookup("ttm_on_epoch_begin");
		plugin.fnBatchBegin = lookup("ttm_on_batch_begin");
		plugin.fnLossComputed = lookup("ttm_on_loss_computed");
		plugin.fnBatchEnd = lookup("ttm_on_batch_end");
		plugin.fnEpochEnd = lookup("ttm_on_epoch_end");
		plugin.fnValidationEnd = lookup("ttm_on_validation_end");
		plugin.fnFitEnd = lookup("ttm_on_fit_end");
		plugin.fnTeardown = lookup("ttm_plugin_teardown");

		return {};
	}

	void wasm_loader_unload(Plugin& plugin) {
		if (plugin.env != nullptr && plugin.fnTeardown != nullptr) {
			wasm_runtime_call_wasm(plugin.env, plugin.fnTeardown, 0, nullptr);
		}
		if (plugin.env != nullptr) {
			wasm_runtime_destroy_exec_env(plugin.env);
			plugin.env = nullptr;
		}
		if (plugin.inst != nullptr) {
			wasm_runtime_deinstantiate(plugin.inst);
			plugin.inst = nullptr;
		}
		if (plugin.module != nullptr) {
			wasm_runtime_unload(plugin.module);
			plugin.module = nullptr;
		}
	}

} // namespace ttm::plugins

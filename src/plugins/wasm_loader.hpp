/**
 * @file wasm_loader.hpp
 * @brief Internal WAMR-based plugin loader types and helpers.
 *
 * @details
 * This header is private to the `ttm_plugins` library and is not installed.
 * It declares the Plugin struct that PluginManager stores per loaded WASM
 * module, and the helper functions used to load and tear down plugins.
 *
 * All functions that can fail return `std::expected<T, std::string>` rather
 * than throwing exceptions, so callers must explicitly handle error paths.
 *
 * @note All WAMR API calls are confined to wasm_loader.cpp so that the rest
 *       of the library does not need to include WAMR headers directly.
 */

#pragma once

#include <ttm/plugins/abi.h>
#include <ttm/plugins/extension.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <ttm/compat/expected.hpp>
#include <vector>

/* WAMR public headers */
#include <wasm_export.h>

namespace ttm::plugins {

	/* Forward declaration — PluginManager owns Plugin instances. */
	class PluginManager;

	/**
	 * @brief Per-plugin runtime state.
	 *
	 * @details
	 * Holds all WAMR handles for a loaded WASM module as well as the resolved
	 * optional lifecycle hook function pointers and the C++ extension objects
	 * (e.g. IDatasetSource instances) registered during ttm_plugin_init.
	 *
	 * Lifecycle:
	 * - Created by wasm_loader_load().
	 * - Destroyed by wasm_loader_unload(), which calls ttm_plugin_teardown and
	 *   releases all WAMR handles.
	 *
	 * @see wasm_loader_load
	 * @see wasm_loader_unload
	 */
	struct Plugin {
		/* -----------------------------------------------------------------
		 * WAMR runtime handles
		 * -------------------------------------------------------------- */
		wasm_module_t module = nullptr;	   ///< Loaded WASM module.
		wasm_module_inst_t inst = nullptr; ///< Module instance.
		wasm_exec_env_t env = nullptr;	   ///< Execution environment.

		/**
		 * @brief Persistent copy of ttm_host_api used as WAMR user-data.
		 *
		 * @details
		 * The host_api pointer stored via wasm_runtime_set_user_data must
		 * remain valid for the entire lifetime of the plugin.  Copying it
		 * into the Plugin struct (which lives in PluginManager::plugins)
		 * ensures it is never dangling when lifecycle callbacks fire.
		 */
		ttm_host_api persistentApi{};

		/**
		 * @brief Back-reference to the owning PluginManager.
		 *
		 * @details
		 * Stored alongside persistentApi so that host imports that need to
		 * route calls back through the manager (e.g. host_log_metric) can
		 * do so without relying on the registration-time ctx pointer, which
		 * is only valid during ttm_plugin_init.
		 */
		PluginManager* manager = nullptr;

		/* -----------------------------------------------------------------
		 * Resolved optional lifecycle hooks (nullptr = not exported)
		 * -------------------------------------------------------------- */
		wasm_function_inst_t fnFitBegin = nullptr;		///< @see ttm_on_fit_begin
		wasm_function_inst_t fnEpochBegin = nullptr;	///< @see ttm_on_epoch_begin
		wasm_function_inst_t fnBatchBegin = nullptr;	///< @see ttm_on_batch_begin
		wasm_function_inst_t fnLossComputed = nullptr;	///< @see ttm_on_loss_computed
		wasm_function_inst_t fnBatchEnd = nullptr;		///< @see ttm_on_batch_end
		wasm_function_inst_t fnEpochEnd = nullptr;		///< @see ttm_on_epoch_end
		wasm_function_inst_t fnValidationEnd = nullptr; ///< @see ttm_on_validation_end
		wasm_function_inst_t fnFitEnd = nullptr;		///< @see ttm_on_fit_end
		wasm_function_inst_t fnOnLog = nullptr;			///< @see ttm_on_log
		wasm_function_inst_t fnOnMetric = nullptr;		///< @see ttm_on_metric
		wasm_function_inst_t fnTeardown = nullptr;		///< @see ttm_plugin_teardown

		/* -----------------------------------------------------------------
		 * C++ extension objects registered by this plugin
		 * -------------------------------------------------------------- */

		/**
		 * @brief Dataset sources registered by this plugin.
		 * @details Ownership lives here; PluginManager::sourceRegistry holds
		 *          non-owning raw pointers.
		 */
		std::vector<std::unique_ptr<IDatasetSource>> sources;

		/**
		 * @brief ML tasks registered by this plugin.
		 * @details Ownership lives here; PluginManager::taskRegistry holds
		 *          non-owning raw pointers.
		 */
		std::vector<std::unique_ptr<ITask>> tasks;

		/**
		 * @brief Model loaders registered by this plugin.
		 */
		std::vector<std::unique_ptr<IModelLoader>> modelLoaders;

		/**
		 * @brief Transforms registered by this plugin.
		 */
		std::vector<std::unique_ptr<ITransform>> transforms;
	};

	/* =========================================================================
	 * Loader functions
	 * ====================================================================== */

	/**
	 * @brief Initialise the WAMR runtime (refcounted — safe to call multiple times).
	 *
	 * @details
	 * The runtime is reference-counted so that multiple PluginManager instances
	 * (e.g. in unit tests) are safe.  Each successful call to wasm_loader_init
	 * must be paired with exactly one call to wasm_loader_destroy.
	 *
	 * @return `{}` on success, or an error string on failure.
	 */
	[[nodiscard]] std::expected<void, std::string> wasm_loader_init();

	/**
	 * @brief Decrement the WAMR runtime reference count.
	 *
	 * @details Shuts down the runtime when the last reference is released.
	 * Always succeeds.
	 *
	 * @see wasm_loader_init
	 */
	void wasm_loader_destroy();

	/**
	 * @brief Load, instantiate, and initialise a WASM plugin.
	 *
	 * @details
	 * Loading sequence:
	 * 1. Read the `.wasm` file from disk.
	 * 2. Compile via `wasm_runtime_load()`.
	 * 3. Register host import functions under the `"ttm"` module namespace.
	 * 4. Instantiate via `wasm_runtime_instantiate()`.
	 * 5. Call `ttm_plugin_get_info`; check ABI version.
	 * 6. Build a #ttm_host_api and call `ttm_plugin_init`.
	 * 7. Resolve optional lifecycle hook pointers.
	 *
	 * On failure the plugin struct is left fully cleaned up (no dangling handles).
	 *
	 * @param[in]  path         Path to the `.wasm` file.
	 * @param[in]  config_json  JSON configuration string passed to `ttm_plugin_init`.
	 * @param[in]  host_api     Fully populated host API struct (callbacks + ctx).
	 * @param[out] plugin       Plugin struct to populate on success.
	 *
	 * @return `{}` on success, or an error string describing which step failed.
	 *
	 * @see wasm_loader_unload
	 */
	[[nodiscard]] std::expected<void, std::string> wasm_loader_load(
			const std::filesystem::path& path, std::string_view config_json, const ttm_host_api& host_api,
			Plugin& plugin
	);

	/**
	 * @brief Tear down a plugin and release all WAMR handles.
	 *
	 * @details
	 * Calls `ttm_plugin_teardown` (if exported), then destroys the execution
	 * environment, module instance, and module in that order.  Always succeeds.
	 *
	 * @param[in,out] plugin  Plugin to unload; all handles are set to nullptr on return.
	 *
	 * @see wasm_loader_load
	 */
	void wasm_loader_unload(Plugin& plugin);

	/* =========================================================================
	 * WASM memory helpers
	 * ====================================================================== */

	/**
	 * @brief Copy a string into WASM linear memory and return (wasm_ptr, len).
	 *
	 * @details
	 * Allocates `str.size()` bytes in the plugin's linear memory via
	 * `wasm_runtime_module_malloc`, writes the bytes with `memcpy`, and sets the
	 * output parameters.  The caller is responsible for freeing the allocation
	 * via `wasm_runtime_module_free`.
	 *
	 * @param[in]  inst     WASM module instance that owns the linear memory.
	 * @param[in]  str      String to copy (not required to be NUL-terminated).
	 * @param[out] wasm_ptr Set to the WASM-side linear-memory offset on success.
	 * @param[out] len      Set to the number of bytes copied (== str.size()).
	 *
	 * @return `{}` on success, or an error string if WASM-heap allocation fails.
	 */
	[[nodiscard]] std::expected<void, std::string>
	wasm_push_string(wasm_module_inst_t inst, std::string_view str, uint32_t& wasm_ptr, uint32_t& len);

} // namespace ttm::plugins

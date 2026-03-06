/**
 * @file wasm_loader.hpp
 * @brief Internal WAMR-based plugin loader types and helpers.
 *
 * @details
 * This header is private to the `ttm_plugins` library and is not installed.
 * It declares the Plugin struct that PluginManager stores per loaded WASM
 * module, and the helper functions used to load and tear down plugins.
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
    wasm_module_t      module  = nullptr; ///< Loaded WASM module.
    wasm_module_inst_t inst    = nullptr; ///< Module instance.
    wasm_exec_env_t    env     = nullptr; ///< Execution environment.

    /* -----------------------------------------------------------------
     * Resolved optional lifecycle hooks (nullptr = not exported)
     * -------------------------------------------------------------- */
    wasm_function_inst_t fn_fit_begin      = nullptr; ///< @see ttm_on_fit_begin
    wasm_function_inst_t fn_epoch_begin    = nullptr; ///< @see ttm_on_epoch_begin
    wasm_function_inst_t fn_batch_begin    = nullptr; ///< @see ttm_on_batch_begin
    wasm_function_inst_t fn_loss_computed  = nullptr; ///< @see ttm_on_loss_computed
    wasm_function_inst_t fn_batch_end      = nullptr; ///< @see ttm_on_batch_end
    wasm_function_inst_t fn_epoch_end      = nullptr; ///< @see ttm_on_epoch_end
    wasm_function_inst_t fn_validation_end = nullptr; ///< @see ttm_on_validation_end
    wasm_function_inst_t fn_fit_end        = nullptr; ///< @see ttm_on_fit_end
    wasm_function_inst_t fn_teardown       = nullptr; ///< @see ttm_plugin_teardown

    /* -----------------------------------------------------------------
     * C++ extension objects registered by this plugin
     * -------------------------------------------------------------- */

    /**
     * @brief Dataset sources registered by this plugin.
     * @details Ownership lives here; PluginManager::source_registry_ holds
     *          non-owning raw pointers.
     */
    std::vector<std::unique_ptr<IDatasetSource>> sources;
};

/* =========================================================================
 * Loader functions
 * ====================================================================== */

/**
 * @brief Initialise the WAMR runtime.
 *
 * @details Must be called once before any wasm_loader_load() calls.
 * @throws std::runtime_error on failure.
 */
void wasm_loader_init();

/**
 * @brief Shut down the WAMR runtime.
 *
 * @details Must be called after all plugins have been unloaded.
 * @see wasm_loader_unload
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
 * @param[in]  path         Path to the `.wasm` file.
 * @param[in]  config_json  JSON configuration string passed to `ttm_plugin_init`.
 * @param[in]  host_api     Fully populated host API struct (callbacks + ctx).
 * @param[out] plugin       Plugin struct to populate; caller owns the result.
 *
 * @throws std::runtime_error if any step fails.
 *
 * @see wasm_loader_unload
 */
void wasm_loader_load(const std::filesystem::path& path,
                      std::string_view config_json,
                      const ttm_host_api& host_api,
                      Plugin& plugin);

/**
 * @brief Tear down a plugin and release all WAMR handles.
 *
 * @details
 * Calls `ttm_plugin_teardown` (if exported), then destroys the execution
 * environment, module instance, and module in that order.
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
 * `wasm_runtime_module_malloc`, writes the bytes with `memcpy`, and returns
 * the WASM-side pointer and byte count.  The caller is responsible for
 * freeing the allocation (via the ABI free callback or directly).
 *
 * @param[in]  inst     WASM module instance that owns the linear memory.
 * @param[in]  str      String to copy (not required to be NUL-terminated).
 * @param[out] wasm_ptr Set to the WASM-side linear-memory offset.
 * @param[out] len      Set to the number of bytes copied (== str.size()).
 *
 * @throws std::runtime_error if allocation in WASM linear memory fails.
 */
void wasm_push_string(wasm_module_inst_t inst,
                      std::string_view str,
                      uint32_t& wasm_ptr,
                      uint32_t& len);

} // namespace ttm::plugins

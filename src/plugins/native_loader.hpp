/**
 * @file native_loader.hpp
 * @brief Native shared-library plugin loader.
 *
 * @details
 * Complements wasm_loader.hpp for plugins compiled as native shared libraries
 * (.so / .dylib / .dll).  Native plugins have direct access to the host
 * process and its libraries (e.g. libgit2) which cannot easily be compiled to
 * WASM.
 *
 * The loading sequence mirrors wasm_loader_load():
 * 1. dlopen / LoadLibrary
 * 2. Resolve required exports (ttm_plugin_get_info, ttm_plugin_init)
 * 3. Verify ABI version
 * 4. Call ttm_plugin_init with a real ttm_host_api (all function pointers are
 *    valid host addresses — no WASM linear-memory translation needed)
 * 5. Resolve optional lifecycle hooks
 *
 * @note This header is private to the ttm_plugins library.
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

namespace ttm::plugins {

	/**
	 * @brief Per-plugin runtime state for native shared-library plugins.
	 *
	 * @details
	 * Holds the dynamic-library handle, resolved entry points, and the C++
	 * extension objects (sources, tasks) registered during ttm_plugin_init.
	 *
	 * Lifecycle:
	 * - Populated by native_loader_load().
	 * - Destroyed by native_loader_unload(), which calls ttm_plugin_teardown
	 *   (if present) and then closes the library handle.
	 */
	struct NativePlugin {
		/* -----------------------------------------------------------------
		 * Library handle
		 * -------------------------------------------------------------- */
		void* dlHandle = nullptr; ///< dlopen / LoadLibrary handle.

		/* -----------------------------------------------------------------
		 * Persistent host API copy
		 *
		 * native_loader_load() copies the caller's ttm_host_api here before
		 * passing &persistentApi to ttm_plugin_init, so that native plugins
		 * may safely retain the pointer for post-init use.  ctx is initially
		 * the PluginRegistrationCtx* for registration; PluginManager::load()
		 * updates it to PluginManager* after a successful load so lifecycle
		 * callbacks receive a valid context.
		 * -------------------------------------------------------------- */
		ttm_host_api persistentApi{};

		/* -----------------------------------------------------------------
		 * Required entry points (non-null after successful load)
		 * -------------------------------------------------------------- */
		ttm_plugin_info* (*fnGetInfo)()                                        = nullptr;
		ttm_error        (*fnInit)(const ttm_host_api*, const char*, uint32_t) = nullptr;

		/* -----------------------------------------------------------------
		 * Optional entry points (nullptr = not exported by this plugin)
		 * -------------------------------------------------------------- */
		void    (*fnTeardown)()                                                       = nullptr;
		void    (*fnFitBegin)(const char*, uint32_t)                                  = nullptr;
		void    (*fnEpochBegin)(uint32_t, uint32_t)                                   = nullptr;
		void    (*fnBatchBegin)(uint32_t, uint32_t)                                   = nullptr;
		float   (*fnLossComputed)(float)                                              = nullptr;
		void    (*fnBatchEnd)(uint32_t, float, const char*, uint32_t)                 = nullptr;
		int32_t (*fnEpochEnd)(uint32_t, const char*, uint32_t)                        = nullptr;
		void    (*fnValidationEnd)(const char*, uint32_t)                             = nullptr;
		void    (*fnFitEnd)(const char*, uint32_t)                                    = nullptr;
		void    (*fnOnLog)(uint32_t level, const char* msg, uint32_t len)              = nullptr;
		void    (*fnOnMetric)(const char* key, uint32_t key_len, float v, int32_t step) = nullptr;

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
		 * @details Ownership lives here; PluginManager::modelLoaderRegistry holds
		 *          non-owning raw pointers.
		 */
		std::vector<std::unique_ptr<IModelLoader>> modelLoaders;

		/**
		 * @brief Transforms (preprocessors) registered by this plugin.
		 * @details Ownership lives here; PluginManager::transformRegistry holds
		 *          non-owning raw pointers.
		 */
		std::vector<std::unique_ptr<ITransform>> transforms;

		/** @brief Optional: called when a model is successfully loaded. */
		void (*fnOnModelLoaded)(const char* info_json, uint32_t len) = nullptr;
	};

	/* =========================================================================
	 * Loader functions
	 * ====================================================================== */

	/**
	 * @brief Load, initialise, and resolve a native shared-library plugin.
	 *
	 * @param[in]  path         Path to the shared library (.so / .dylib / .dll).
	 * @param[in]  config_json  JSON configuration string passed to ttm_plugin_init.
	 * @param[in]  host_api     Fully populated host API struct (callbacks + ctx).
	 * @param[out] plugin       NativePlugin struct to populate on success.
	 *
	 * @return `{}` on success, or an error string describing which step failed.
	 */
	[[nodiscard]] std::expected<void, std::string> native_loader_load(
			const std::filesystem::path& path, std::string_view config_json,
			const ttm_host_api& host_api, NativePlugin& plugin
	);

	/**
	 * @brief Tear down a native plugin and unload the library.
	 *
	 * @details
	 * Calls ttm_plugin_teardown (if present) then closes the library handle.
	 * Always succeeds.
	 *
	 * @param[in,out] plugin  Plugin to unload; dlHandle is set to nullptr on return.
	 */
	void native_loader_unload(NativePlugin& plugin);

} // namespace ttm::plugins

/**
 * @file native_loader.cpp
 * @brief Native shared-library plugin loader implementation.
 *
 * @details
 * Uses dlopen / dlsym / dlclose on POSIX and LoadLibrary / GetProcAddress /
 * FreeLibrary on Windows to load native plugins (.so / .dylib / .dll).
 *
 * All platform-specific code is guarded with #ifdef _WIN32.
 */

#include "native_loader.hpp"

#include <ttm/plugins/abi.h>

#include <cstdint>
#include <filesystem>
#include <ttm/compat/format.hpp>
#include <string>
#include <string_view>
#include <ttm/compat/expected.hpp>

#ifdef _WIN32
#	define WIN32_LEAN_AND_MEAN
#	include <windows.h>
#else
#	include <dlfcn.h>
#endif

namespace ttm::plugins {

	namespace {

		/* ------------------------------------------------------------------
		 * Platform helpers
		 * ---------------------------------------------------------------- */

		void* lib_open(const std::filesystem::path& path) {
#ifdef _WIN32
			return static_cast<void*>(LoadLibraryW(path.wstring().c_str()));
#else
			return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
		}

		void* lib_sym(void* handle, const char* name) {
#ifdef _WIN32
			// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- Windows API
			return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
			return dlsym(handle, name);
#endif
		}

		void lib_close(void* handle) {
			if (handle == nullptr) {
				return;
			}
#ifdef _WIN32
			FreeLibrary(static_cast<HMODULE>(handle));
#else
			dlclose(handle);
#endif
		}

		std::string lib_error() {
#ifdef _WIN32
			const DWORD err = GetLastError();
			char buf[256]{};
			FormatMessageA(
					FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, err,
					MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, sizeof(buf), nullptr
			);
			return buf;
#else
			const char* msg = dlerror();
			return msg != nullptr ? msg : "(unknown dl error)";
#endif
		}

		/* ------------------------------------------------------------------
		 * Type-safe symbol resolution
		 * ---------------------------------------------------------------- */
		template<typename Fn>
		Fn resolve(void* handle, const char* name) {
			void* sym = lib_sym(handle, name);
			Fn fn     = nullptr;
			// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- required for dlsym portability
			__builtin_memcpy(&fn, &sym, sizeof(fn));
			return fn;
		}

	} // anonymous namespace

	/* =========================================================================
	 * native_loader_load
	 * ====================================================================== */

	std::expected<void, std::string> native_loader_load(
			const std::filesystem::path& path, std::string_view config_json,
			const ttm_host_api& host_api, NativePlugin& plugin
	) {
		/* 1. Open library --------------------------------------------------- */
		plugin.dlHandle = lib_open(path);
		if (plugin.dlHandle == nullptr) {
			return std::unexpected(
					std::format("native_loader_load: cannot open '{}': {}", path.string(), lib_error())
			);
		}

		/* 2. Resolve required symbols --------------------------------------- */
		plugin.fnGetInfo = resolve<ttm_plugin_info* (*)()>(plugin.dlHandle, "ttm_plugin_get_info");
		if (plugin.fnGetInfo == nullptr) {
			lib_close(plugin.dlHandle);
			plugin.dlHandle = nullptr;
			return std::unexpected(
					std::format("native_loader_load: '{}' does not export ttm_plugin_get_info", path.string())
			);
		}

		plugin.fnInit =
				resolve<ttm_error (*)(const ttm_host_api*, const char*, uint32_t)>(plugin.dlHandle, "ttm_plugin_init");
		if (plugin.fnInit == nullptr) {
			lib_close(plugin.dlHandle);
			plugin.dlHandle = nullptr;
			return std::unexpected(
					std::format("native_loader_load: '{}' does not export ttm_plugin_init", path.string())
			);
		}

		/* 3. Verify ABI version --------------------------------------------- */
		const ttm_plugin_info* info = plugin.fnGetInfo();
		if (info == nullptr) {
			lib_close(plugin.dlHandle);
			plugin.dlHandle = nullptr;
			return std::unexpected("native_loader_load: ttm_plugin_get_info returned null");
		}
		if (info->abiVersion != TTM_ABI_VERSION) {
			lib_close(plugin.dlHandle);
			plugin.dlHandle = nullptr;
			return std::unexpected(std::format(
					"native_loader_load: plugin ABI version mismatch (plugin={}, host={})",
					info->abiVersion, TTM_ABI_VERSION
			));
		}

		/* 4. Call ttm_plugin_init ------------------------------------------- */
		const auto config_str = std::string(config_json);
		const auto init_result =
				plugin.fnInit(&host_api, config_str.c_str(), static_cast<uint32_t>(config_str.size()));
		if (init_result != TTM_OK) {
			lib_close(plugin.dlHandle);
			plugin.dlHandle = nullptr;
			return std::unexpected(
					std::format("native_loader_load: ttm_plugin_init returned error {}", static_cast<int>(init_result))
			);
		}

		/* 5. Resolve optional lifecycle hooks -------------------------------- */
		plugin.fnTeardown      = resolve<void (*)()>(plugin.dlHandle, "ttm_plugin_teardown");
		plugin.fnFitBegin      = resolve<void (*)(const char*, uint32_t)>(plugin.dlHandle, "ttm_on_fit_begin");
		plugin.fnEpochBegin    = resolve<void (*)(uint32_t, uint32_t)>(plugin.dlHandle, "ttm_on_epoch_begin");
		plugin.fnBatchBegin    = resolve<void (*)(uint32_t, uint32_t)>(plugin.dlHandle, "ttm_on_batch_begin");
		plugin.fnLossComputed  = resolve<float (*)(float)>(plugin.dlHandle, "ttm_on_loss_computed");
		plugin.fnBatchEnd      = resolve<void (*)(uint32_t, float, const char*, uint32_t)>(plugin.dlHandle, "ttm_on_batch_end");
		plugin.fnEpochEnd      = resolve<int32_t (*)(uint32_t, const char*, uint32_t)>(plugin.dlHandle, "ttm_on_epoch_end");
		plugin.fnValidationEnd = resolve<void (*)(const char*, uint32_t)>(plugin.dlHandle, "ttm_on_validation_end");
		plugin.fnFitEnd        = resolve<void (*)(const char*, uint32_t)>(plugin.dlHandle, "ttm_on_fit_end");

		return {};
	}

	/* =========================================================================
	 * native_loader_unload
	 * ====================================================================== */

	void native_loader_unload(NativePlugin& plugin) {
		if (plugin.fnTeardown != nullptr) {
			plugin.fnTeardown();
			plugin.fnTeardown = nullptr;
		}
		lib_close(plugin.dlHandle);
		plugin.dlHandle = nullptr;
	}

} // namespace ttm::plugins

/**
 * @file plugin_ctx.hpp
 * @brief PluginRegistrationCtx — context passed to plugin host-API callbacks.
 *
 * @details
 * Both WASM and native plugins receive a ttm_host_api whose ctx field points
 * to a PluginRegistrationCtx.  This struct lets the static host callbacks
 * (s_register_source, s_register_task, …) know both the owning PluginManager
 * and which plugin record should receive newly-registered extension objects.
 *
 * The struct is stack-allocated inside PluginManager::load() and is valid
 * for the duration of the ttm_plugin_init call only.
 */

#pragma once

#include <ttm/plugins/extension.hpp>

#include <functional>
#include <memory>

namespace ttm::plugins {

	struct Plugin;
	struct NativePlugin;
	class PluginManager;

	/**
	 * @brief Context threaded through ttm_host_api.ctx during plugin init.
	 *
	 * Exactly one of wasmPlugin / nativePlugin is non-null, depending on which
	 * loader is being used.
	 *
	 * @details
	 * The attach_source and attach_task callbacks are set by
	 * PluginManager::make_host_api() and delegate to the private
	 * register_source_impl / register_task_impl methods.  This avoids the need
	 * for WASM loader code (which cannot include WAMR headers from a public
	 * interface) to access private PluginManager methods directly.
	 */
	struct PluginRegistrationCtx {
		PluginManager* manager     = nullptr;
		Plugin*        wasmPlugin   = nullptr; ///< Set for WASM plugins.
		NativePlugin*  nativePlugin = nullptr; ///< Set for native plugins.

		/** @brief Transfer ownership of a source into the owning plugin record + registry. */
		std::function<void(std::unique_ptr<IDatasetSource>)> attach_source;

		/** @brief Transfer ownership of a task into the owning plugin record + registry. */
		std::function<void(std::unique_ptr<ITask>)> attach_task;

		/** @brief Transfer ownership of a model loader into the owning plugin record + registry. */
		std::function<void(std::unique_ptr<IModelLoader>)> attach_model_loader;

		/** @brief Transfer ownership of a transform into the owning plugin record + registry. */
		std::function<void(std::unique_ptr<ITransform>)> attach_transform;
	};

} // namespace ttm::plugins

/**
 * @file plugin_manager.hpp
 * @brief PluginManager — loads WASM plugins and dispatches lifecycle events.
 *
 * @details
 * PluginManager is the central runtime registry for all TMM plugin extensions.
 * It owns the WAMR runtime and all plugin instances.  A single PluginManager
 * is expected to live for the duration of the process.
 *
 * @par Typical usage
 * @code{.cpp}
 * auto mgr = tmm::plugins::PluginManager::create().value();
 *
 * // Load plugins declared in the project config
 * mgr.load("plugins/my-source.wasm", R"({"token":"..."})").value();
 *
 * // Resolve a dataset source and open a URI
 * auto* src = mgr.findSource("gh:");
 * auto reader = src->open("gh:owner/repo/train.jsonl");
 *
 * // Training loop
 * mgr.emitFitBegin(ctx_json);
 * for (uint32_t ep = 0; ep < epochs; ++ep) {
 *     mgr.emitEpochBegin(ep, epochs);
 *     for (uint32_t b = 0; b < batches; ++b) {
 *         mgr.emitBatchBegin(b, batches);
 *         float loss = compute_loss(…);
 *         loss = mgr.emitLossComputed(loss);
 *         mgr.emitBatchEnd(b, loss, metrics_json);
 *     }
 *     if (mgr.emitEpochEnd(ep, metrics_json)) break; // early stop
 * }
 * mgr.emitFitEnd(final_metrics_json);
 * @endcode
 *
 * @see tmm::plugins::IDatasetSource  Extension point for URI-based data loading
 * @see abi.h                         C ABI that plugins implement
 */

#ifndef TMM_PLUGINS_PLUGIN_MANAGER_HPP
#define TMM_PLUGINS_PLUGIN_MANAGER_HPP

#include <tmm/plugins/abi.h>
#include <tmm/plugins/extension.hpp>
#include <tmm/trainer/callback.hpp>
#include <tmm/trainer/interfaces.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <tmm/compat/expected.hpp>
#include <unordered_map>
#include <vector>

namespace tmm::plugins {

	/**
	 * @brief Central manager for TMM plugins.
	 *
	 * @details
	 * Responsibilities:
	 * - Initialise and own the WAMR runtime.
	 * - Load WASM plugin modules from disk (load()).
	 * - Maintain registries of extension objects (sources, transforms, tasks, metrics).
	 * - Dispatch lifecycle events (emit*()) to all loaded plugins.
	 *
	 * @note Non-copyable; movable.  After a move the source object is left in a
	 *       safe but empty state and must not be used further.
	 *
	 * @see load           Load a plugin from a WASM file
	 * @see findSource     Look up a registered data source by URI scheme
	 */
	class PluginManager {
	public:
		/**
		 * @brief Named constructor — creates a PluginManager and registers the
		 *        built-in `file:` source.
		 *
		 * @return A fully initialised PluginManager, or an error string if the
		 *         WAMR runtime cannot be initialised.
		 */
		[[nodiscard]] static std::expected<PluginManager, std::string> create();

		/**
		 * @brief Destroy all plugins (calling tmm_plugin_teardown on each) and
		 *        shut down the WAMR runtime.
		 */
		~PluginManager();

		PluginManager(const PluginManager&) = delete;
		PluginManager& operator=(const PluginManager&) = delete;

		/** @brief Move constructor — transfers ownership of WAMR runtime and plugins. */
		PluginManager(PluginManager&&) noexcept;
		/** @brief Move assignment — transfers ownership of WAMR runtime and plugins. */
		PluginManager& operator=(PluginManager&&) noexcept;

		/* =====================================================================
		 * @defgroup pm_loading Plugin loading
		 * @{
		 * ================================================================== */

		/**
		 * @brief Load a WASM plugin from disk and initialise it.
		 *
		 * @details
		 * Loading sequence:
		 * 1. Read the `.wasm` file into memory.
		 * 2. Compile/load via `wasm_runtime_load()`.
		 * 3. Register host import functions (`tmm` module namespace).
		 * 4. Instantiate via `wasm_runtime_instantiate()`.
		 * 5. Verify ABI version by calling `tmm_plugin_get_info`.
		 * 6. Call `tmm_plugin_init(host_api, config_json, config_len)`.
		 * 7. Resolve optional lifecycle hook function pointers.
		 *
		 * @param[in] path         Path to the `.wasm` file.
		 * @param[in] config_json  Plugin-specific JSON configuration passed verbatim
		 *                         to `tmm_plugin_init()`.  Pass `"{}"` (the default)
		 *                         if the plugin requires no configuration.
		 *
		 * @return `{}` on success, or an error string if any loading step fails.
		 *
		 * @par Example
		 * @code{.cpp}
		 * mgr.load("plugins/hf-source.wasm").value();
		 * mgr.load("plugins/bpe-tokenizer.wasm", R"({"vocab":"bpe.json"})").value();
		 * @endcode
		 *
		 * @see tmm_plugin_get_info  ABI entry-point queried in step 5
		 * @see tmm_plugin_init      ABI entry-point called in step 6
		 */
		[[nodiscard]] std::expected<void, std::string>
		load(const std::filesystem::path& path, std::string_view config_json = "{}");

		/** @} */

		/* =====================================================================
		 * @defgroup pm_registry Extension registries
		 * @{
		 * ================================================================== */

		/**
		 * @brief Look up a registered dataset source by URI scheme.
		 *
		 * @details
		 * The scheme must include the trailing colon, e.g. `"file:"`, `"gh:"`.
		 * The built-in `"file:"` source is always available after construction.
		 *
		 * @param[in] scheme  URI scheme string (e.g. `"hf:"`, `"file:"`).
		 * @return Non-owning pointer to the source, or `nullptr` if no plugin
		 *         handles the requested scheme.  Valid for the lifetime of
		 *         this PluginManager.
		 *
		 * @see IDatasetSource
		 */
		[[nodiscard]] IDatasetSource* findSource(std::string_view scheme) const;

		/**
		 * @brief Look up a registered ML task by canonical name or alias.
		 *
		 * @param[in] name_or_alias  Task name or alias (e.g. `"text-classification"`, `"tc"`).
		 * @return Non-owning pointer to the task, or `nullptr` if not found.
		 *         Valid for the lifetime of this PluginManager.
		 *
		 * @see ITask
		 */
		[[nodiscard]] ITask* findTask(std::string_view name_or_alias) const;

		/**
		 * @brief Find a registered model loader that accepts the given file path.
		 *
		 * @details
		 * Iterates all registered IModelLoader instances in registration order
		 * and returns the first one whose probe() method returns true for
		 * `path`.  Returns `nullptr` if no loader claims the file.
		 *
		 * @param[in] path  Model file path (e.g. `"./gpt2.so"`, `"model.py"`).
		 * @return Non-owning pointer to the loader, or `nullptr` if not found.
		 *         Valid for the lifetime of this PluginManager.
		 *
		 * @see IModelLoader
		 */
		[[nodiscard]] IModelLoader* findModelLoader(std::string_view path) const;

		/**
		 * @brief Look up a registered transform (preprocessor) by name.
		 *
		 * @param[in] name  Transform name registered via register_transform.
		 * @return Non-owning pointer to the transform, or `nullptr` if not found.
		 */
		[[nodiscard]] ITransform* findTransform(std::string_view name) const;

		/**
		 * @brief Look up a registered transform vtable by name.
		 *
		 * @details
		 * Unlike findTransform(), which returns a pre-created instance with
		 * empty config, this returns the raw vtable so callers can create new
		 * instances with per-invocation configuration.
		 *
		 * @param[in] name  Transform name registered via register_transform.
		 * @return Non-owning pointer to the vtable copy, or `nullptr` if not found.
		 *         Valid for the lifetime of this PluginManager.
		 */
		[[nodiscard]] const tmm_transform_vtable* findTransformVtable(std::string_view name) const;

		/**
		 * @brief Broadcast model-loaded metadata to all plugins that export
		 *        #tmm_on_model_loaded.
		 *
		 * @param[in] info_json  JSON-serialised model info string.
		 */
		void emitModelLoaded(std::string_view info_json);

		/**
		 * @brief Look up a registered LR scheduler vtable by name.
		 *
		 * @details
		 * Schedulers are registered by plugins via `host->register_scheduler()`.
		 * The built-in schedulers (constant, step, linear, cosine, cosine_warmup)
		 * are provided by the core native plugin.
		 *
		 * @param[in] name  Scheduler name from the training config (e.g. "cosine_warmup").
		 * @return Non-owning pointer to the vtable copy, or `nullptr` if not found.
		 *         Valid for the lifetime of this PluginManager.
		 *
		 * @see tmm_scheduler_vtable
		 */
		[[nodiscard]] const tmm_scheduler_vtable* findSchedulerVtable(std::string_view name) const;

		/**
		 * @brief Look up a registered optimizer vtable by name.
		 *
		 * @param[in] name  Optimizer name from the training config (e.g. "adamw").
		 * @return Non-owning pointer to the vtable copy, or `nullptr` if not found.
		 *         Valid for the lifetime of this PluginManager.
		 *
		 * @see tmm_optimizer_vtable
		 */
		[[nodiscard]] const tmm_optimizer_vtable* findOptimizerVtable(std::string_view name) const;

		/**
		 * @brief Look up a registered trainer callback vtable by name.
		 *
		 * @param[in] name  Callback name, optionally qualified (e.g. "early_stopping",
		 *                  "core::early_stopping").
		 * @return Non-owning pointer to the vtable copy, or `nullptr` if not found.
		 *         Valid for the lifetime of this PluginManager.
		 *
		 * @see tmm_trainer_callback_vtable
		 */
		[[nodiscard]] const tmm_trainer_callback_vtable* findCallbackVtable(std::string_view name) const;

		/**
		 * @brief Create a trainer callback and wrap it as a trainer::Callback.
		 *
		 * @param[in]  name        Callback name (e.g. "early_stopping").
		 * @param[in]  config_json JSON config string passed to vtable->create().
		 * @param[out] out_error   If non-null, receives an error description on failure.
		 * @return Owning pointer to the callback, or nullptr on failure.
		 */
		[[nodiscard]] std::unique_ptr<tmm::trainer::Callback>
		makeCallback(std::string_view name, std::string_view config_json, std::string* out_error = nullptr) const;

		/**
		 * @brief Create an optimizer and wrap it as a trainer::IOptimizer.
		 *
		 * @details
		 * Looks up the vtable by @p name, calls vtable->create() with the
		 * supplied parameters, and wraps the resulting handle in an adapter
		 * that implements trainer::IOptimizer.  On success the returned
		 * unique_ptr owns the optimizer handle and will destroy it on
		 * destruction.
		 *
		 * @param[in]  name         Optimizer name (e.g. "adamw").
		 * @param[in]  model_h      Model handle (from IModelLoader::open).
		 * @param[in]  params       Host-allocated param DLTensors (may be nullptr).
		 * @param[in]  param_count  Number of param/grad tensor pairs.
		 * @param[in]  grads        Host-allocated grad DLTensors (may be nullptr).
		 * @param[in]  cfg_json     JSON string with optimizer hyperparameters.
		 * @param[out] out_error    If non-null, receives an error description on failure.
		 * @return Owning pointer to the optimizer, or nullptr on failure.
		 */
		[[nodiscard]] std::unique_ptr<tmm::trainer::IOptimizer> makeOptimizer(
				std::string_view name, tmm_handle model_h, const DLTensor* params, uint32_t param_count,
				const DLTensor* grads, std::string_view cfg_json, std::string* out_error = nullptr
		) const;

		/** @} */

		/* =====================================================================
		 * @defgroup pm_lifecycle Lifecycle event dispatch
		 * @{
		 *
		 * These mirror the optional hook functions declared in abi.h.  The manager
		 * iterates all loaded plugins in the order they were loaded and calls the
		 * hook only on plugins that exported it.  For hooks with return values,
		 * results are chained (loss_computed) or accumulated (epoch_end).
		 * ================================================================== */

		/**
		 * @brief Dispatch #tmm_on_fit_begin to all plugins that export it.
		 * @param[in] ctx_json  JSON object carrying run metadata (hyperparameters, etc.).
		 */
		void emitFitBegin(std::string_view ctx_json);

		/**
		 * @brief Dispatch #tmm_on_epoch_begin to all plugins that export it.
		 * @param[in] epoch  0-based current epoch index.
		 * @param[in] total  Total number of planned epochs.
		 */
		void emitEpochBegin(std::uint32_t epoch, std::uint32_t total);

		/**
		 * @brief Dispatch #tmm_on_batch_begin to all plugins that export it.
		 * @param[in] batch  0-based batch index within the current epoch.
		 * @param[in] total  Total batches in the epoch.
		 */
		void emitBatchBegin(std::uint32_t batch, std::uint32_t total);

		/**
		 * @brief Chain the loss value through all plugins that export #tmm_on_loss_computed.
		 *
		 * @details
		 * Each plugin receives the output of the previous one, allowing plugins to
		 * modify (e.g. add regularisation terms) or observe the loss in sequence.
		 *
		 * @param[in] loss  Loss value computed by the training loop.
		 * @return Final loss after all plugins have processed it.
		 */
		float emitLossComputed(float loss);

		/**
		 * @brief Dispatch #tmm_on_batch_end to all plugins that export it.
		 * @param[in] batch        0-based batch index.
		 * @param[in] loss         Final loss for this batch.
		 * @param[in] metrics_json JSON object containing live scalar metrics.
		 */
		void emitBatchEnd(std::uint32_t batch, float loss, std::string_view metrics_json);

		/**
		 * @brief Dispatch #tmm_on_epoch_end to all plugins that export it.
		 *
		 * @details
		 * All plugins are always called so that every plugin sees the epoch end.
		 * The return values are OR-ed across all plugins.
		 *
		 * @param[in] epoch        0-based epoch index.
		 * @param[in] metrics_json JSON object containing epoch-level metrics.
		 * @return `true` if any plugin requested early stopping.
		 */
		bool emitEpochEnd(std::uint32_t epoch, std::string_view metrics_json);

		/**
		 * @brief Dispatch #tmm_on_validation_end to all plugins that export it.
		 * @param[in] metrics_json JSON object containing validation metrics.
		 */
		void emitValidationEnd(std::string_view metrics_json);

		/**
		 * @brief Dispatch #tmm_on_fit_end to all plugins that export it.
		 * @param[in] metrics_json JSON object containing final training metrics.
		 */
		void emitFitEnd(std::string_view metrics_json);

		/**
		 * @brief Broadcast a log message to all plugins that export #tmm_on_log.
		 *
		 * @details
		 * Dispatches to every loaded plugin that exported `tmm_on_log`.  Falls
		 * back to writing to `stderr` if no plugin handles the message, so that
		 * log output is never silently dropped.
		 *
		 * @param[in] level  Severity level.
		 * @param[in] msg    Message text (does not need to be NUL-terminated).
		 */
		void emitLog(tmm_log_level level, std::string_view msg);

		/**
		 * @brief Broadcast a named scalar metric to all plugins that export #tmm_on_metric.
		 *
		 * @details
		 * Dispatches to every loaded plugin except the one that originated the
		 * metric (identified by the calling context).  Used by the Trainer for
		 * PyTorch Lightning–style `self.log("train_loss", loss)` calls.
		 *
		 * @param[in] key    Metric name.
		 * @param[in] value  Scalar value.
		 * @param[in] step   Global training step.
		 */
		void emitMetric(std::string_view key, float value, int32_t step);

		/** @} */

	private:
		/** @brief Private default constructor — use create() instead. */
		PluginManager() = default;

		/** @brief Internal per-plugin state for WASM plugins — defined in plugin_manager.cpp. */
		struct Plugin;

		/** @brief Internal per-plugin state for native plugins — defined in native_loader.hpp. */
		struct NativePlugin;

		/** @brief Loaded WASM plugins, in load order. */
		std::vector<std::unique_ptr<Plugin>> plugins;

		/** @brief Loaded native shared-library plugins, in load order. */
		std::vector<std::unique_ptr<NativePlugin>> nativePlugins;

		/**
		 * @brief Source registry: scheme string → non-owning source pointer.
		 *
		 * Sources are owned by their Plugin / NativePlugin record; this map holds
		 * raw pointers for O(1) scheme lookup.
		 */
		std::unordered_map<std::string, IDatasetSource*> sourceRegistry;

		/**
		 * @brief Task registry: name/alias string → non-owning task pointer.
		 *
		 * Tasks are owned by their Plugin / NativePlugin record; this map holds
		 * raw pointers for O(1) name lookup.  Both canonical names and aliases are
		 * inserted as separate keys mapping to the same ITask*.
		 */
		std::unordered_map<std::string, ITask*> taskRegistry;

		/**
		 * @brief Model loader registry: ordered list of registered loaders.
		 *
		 * @details
		 * findModelLoader() iterates this list in registration order and
		 * returns the first loader whose probe() accepts the given path.
		 * Ownership lives in the owning NativePlugin record.
		 */
		std::vector<IModelLoader*> modelLoaderRegistry;

		/**
		 * @brief Transform registry: name → non-owning transform pointer.
		 */
		std::unordered_map<std::string, ITransform*> transformRegistry;

		/**
		 * @brief Transform vtable registry: name → owned vtable copy.
		 *
		 * @details
		 * Stored by value so callers can create new instances with per-invocation
		 * configuration via findTransformVtable(). The function pointers remain
		 * valid as long as the plugin is loaded.
		 */
		std::unordered_map<std::string, tmm_transform_vtable> transformVtableRegistry;

		/**
		 * @brief Scheduler vtable registry: name → owned vtable copy.
		 *
		 * @details
		 * The vtable struct is copied by value when the plugin registers it.
		 * The function pointers within remain valid as long as the plugin is loaded.
		 */
		std::unordered_map<std::string, tmm_scheduler_vtable> schedulerVtableRegistry;

		/**
		 * @brief Optimizer vtable registry: name → owned vtable copy.
		 * @details The function pointers remain valid as long as the plugin is loaded.
		 */
		std::unordered_map<std::string, tmm_optimizer_vtable> optimizerVtableRegistry;

		/**
		 * @brief Trainer callback vtable registry: name → owned vtable copy.
		 * @details Stores both "name" (first-wins unqualified) and "plugin::name"
		 *          (always stored, always wins for qualified lookup).
		 */
		std::unordered_map<std::string, tmm_trainer_callback_vtable> callbackVtableRegistry;

		/**
		 * @brief True if this instance owns a WAMR runtime reference.
		 *
		 * @details Used by the move constructor/assignment and destructor to ensure
		 *          wasm_loader_destroy() is called exactly once per successful
		 *          wasm_loader_init() call.
		 */
		bool wamrRefOwned = false;

		/** @brief Forward declaration — full type in plugin_ctx.hpp. */
		struct PluginRegistrationCtx;

		/* -----------------------------------------------------------------
		 * Static host API callbacks (ctx == PluginRegistrationCtx*)
		 * -------------------------------------------------------------- */

		/// @private
		static tmm_error s_register_model_loader(void* ctx, const tmm_model_loader_vtable* vt);
		/// @private
		static void s_notify_model_info(void* ctx, const tmm_model_info_t* info);
		/// @private
		static tmm_error s_register_scheduler(void* ctx, const char* name, const tmm_scheduler_vtable* vt);
		/// @private
		static tmm_error s_register_optimizer(void* ctx, const char* name, const tmm_optimizer_vtable* vt);
		/// @private
		static tmm_error s_register_callback(void* ctx, const char* name, const tmm_trainer_callback_vtable* vt);
		/// @private
		void register_model_loader_impl(std::unique_ptr<IModelLoader> loader, PluginRegistrationCtx& ctx);
		/// @private
		void register_transform_impl(std::unique_ptr<ITransform> transform, PluginRegistrationCtx& ctx);
		/// @private
		static tmm_error s_register_source(void* ctx, const char** schemes, const tmm_source_vtable* vt);
		/// @private
		static tmm_error
		s_register_transform(void* ctx, const char* name, const char** aliases, const tmm_transform_vtable* vt);
		/// @private
		static tmm_error s_register_task(void* ctx, const char* name, const char** aliases, const tmm_task_vtable* vt);
		/// @private
		static tmm_error
		s_register_metric(void* ctx, const char* name, const char** aliases, const tmm_metric_vtable* vt);
		/// @private
		static void s_log(void* ctx, tmm_log_level level, const char* msg, uint32_t len);
		/// @private
		static void s_log_metric(void* ctx, const char* key, uint32_t key_len, float value, int32_t step);
		/// @private
		static void s_terminal_size(void* ctx, uint32_t* out_width, uint32_t* out_height);
		/// @private
		static void* s_alloc(void* ctx, uint32_t size);
		/// @private
		static void s_free(void* ctx, void* ptr);

		/**
		 * @brief Construct a #tmm_host_api struct pointing to the given ctx.
		 * @param[in] ctx  Registration context for the plugin currently being loaded.
		 * @return Fully populated host API struct.
		 */
		tmm_host_api make_host_api(PluginRegistrationCtx& ctx);

		/**
		 * @brief Register a source object into the scheme registry.
		 * @param[in] src  Source to register (ownership transferred to the owning plugin record).
		 * @param[in] ctx  Registration context identifying which plugin receives ownership.
		 */
		void register_source_impl(std::unique_ptr<IDatasetSource> src, PluginRegistrationCtx& ctx);

		/**
		 * @brief Register a task object into the task registry.
		 * @param[in] task  Task to register (ownership transferred to the owning plugin record).
		 * @param[in] ctx   Registration context identifying which plugin receives ownership.
		 */
		void register_task_impl(std::unique_ptr<ITask> task, PluginRegistrationCtx& ctx);
	};

} // namespace tmm::plugins

#endif /* TMM_PLUGINS_PLUGIN_MANAGER_HPP */

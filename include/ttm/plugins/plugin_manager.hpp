/**
 * @file plugin_manager.hpp
 * @brief PluginManager — loads WASM plugins and dispatches lifecycle events.
 *
 * @details
 * PluginManager is the central runtime registry for all TTM plugin extensions.
 * It owns the WAMR runtime and all plugin instances.  A single PluginManager
 * is expected to live for the duration of the process.
 *
 * @par Typical usage
 * @code{.cpp}
 * auto mgr = ttm::plugins::PluginManager::create().value();
 *
 * // Load plugins declared in the project config
 * mgr.load("plugins/my-source.wasm", R"({"token":"..."})").value();
 *
 * // Resolve a dataset source and open a URI
 * auto* src = mgr.find_source("gh:");
 * auto reader = src->open("gh:owner/repo/train.jsonl");
 *
 * // Training loop
 * mgr.emit_fit_begin(ctx_json);
 * for (uint32_t ep = 0; ep < epochs; ++ep) {
 *     mgr.emit_epoch_begin(ep, epochs);
 *     for (uint32_t b = 0; b < batches; ++b) {
 *         mgr.emit_batch_begin(b, batches);
 *         float loss = compute_loss(…);
 *         loss = mgr.emit_loss_computed(loss);
 *         mgr.emit_batch_end(b, loss, metrics_json);
 *     }
 *     if (mgr.emit_epoch_end(ep, metrics_json)) break; // early stop
 * }
 * mgr.emit_fit_end(final_metrics_json);
 * @endcode
 *
 * @see ttm::plugins::IDatasetSource  Extension point for URI-based data loading
 * @see abi.h                         C ABI that plugins implement
 */

#ifndef TTM_PLUGINS_PLUGIN_MANAGER_HPP
#define TTM_PLUGINS_PLUGIN_MANAGER_HPP

#include <ttm/plugins/abi.h>
#include <ttm/plugins/extension.hpp>

#include <cstdint>
#include <ttm/compat/expected.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ttm::plugins {

	/**
 * @brief Central manager for TTM plugins.
 *
 * @details
 * Responsibilities:
 * - Initialise and own the WAMR runtime.
 * - Load WASM plugin modules from disk (load()).
 * - Maintain registries of extension objects (sources, transforms, tasks, metrics).
 * - Dispatch lifecycle events (emit_*()) to all loaded plugins.
 *
 * @note Non-copyable; movable.  After a move the source object is left in a
 *       safe but empty state and must not be used further.
 *
 * @see load           Load a plugin from a WASM file
 * @see find_source    Look up a registered data source by URI scheme
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
     * @brief Destroy all plugins (calling ttm_plugin_teardown on each) and
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
     * 3. Register host import functions (`ttm` module namespace).
     * 4. Instantiate via `wasm_runtime_instantiate()`.
     * 5. Verify ABI version by calling `ttm_plugin_get_info`.
     * 6. Call `ttm_plugin_init(host_api, config_json, config_len)`.
     * 7. Resolve optional lifecycle hook function pointers.
     *
     * @param[in] path         Path to the `.wasm` file.
     * @param[in] config_json  Plugin-specific JSON configuration passed verbatim
     *                         to `ttm_plugin_init()`.  Pass `"{}"` (the default)
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
     * @see ttm_plugin_get_info  ABI entry-point queried in step 5
     * @see ttm_plugin_init      ABI entry-point called in step 6
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
		[[nodiscard]] IDatasetSource* find_source(std::string_view scheme) const;

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
     * @brief Dispatch #ttm_on_fit_begin to all plugins that export it.
     * @param[in] ctx_json  JSON object carrying run metadata (hyperparameters, etc.).
     */
		void emit_fit_begin(std::string_view ctx_json);

		/**
     * @brief Dispatch #ttm_on_epoch_begin to all plugins that export it.
     * @param[in] epoch  0-based current epoch index.
     * @param[in] total  Total number of planned epochs.
     */
		void emit_epoch_begin(std::uint32_t epoch, std::uint32_t total);

		/**
     * @brief Dispatch #ttm_on_batch_begin to all plugins that export it.
     * @param[in] batch  0-based batch index within the current epoch.
     * @param[in] total  Total batches in the epoch.
     */
		void emit_batch_begin(std::uint32_t batch, std::uint32_t total);

		/**
     * @brief Chain the loss value through all plugins that export #ttm_on_loss_computed.
     *
     * @details
     * Each plugin receives the output of the previous one, allowing plugins to
     * modify (e.g. add regularisation terms) or observe the loss in sequence.
     *
     * @param[in] loss  Loss value computed by the training loop.
     * @return Final loss after all plugins have processed it.
     */
		float emit_loss_computed(float loss);

		/**
     * @brief Dispatch #ttm_on_batch_end to all plugins that export it.
     * @param[in] batch        0-based batch index.
     * @param[in] loss         Final loss for this batch.
     * @param[in] metrics_json JSON object containing live scalar metrics.
     */
		void emit_batch_end(std::uint32_t batch, float loss, std::string_view metrics_json);

		/**
     * @brief Dispatch #ttm_on_epoch_end to all plugins that export it.
     *
     * @details
     * All plugins are always called so that every plugin sees the epoch end.
     * The return values are OR-ed across all plugins.
     *
     * @param[in] epoch        0-based epoch index.
     * @param[in] metrics_json JSON object containing epoch-level metrics.
     * @return `true` if any plugin requested early stopping.
     */
		bool emit_epoch_end(std::uint32_t epoch, std::string_view metrics_json);

		/**
     * @brief Dispatch #ttm_on_validation_end to all plugins that export it.
     * @param[in] metrics_json JSON object containing validation metrics.
     */
		void emit_validation_end(std::string_view metrics_json);

		/**
     * @brief Dispatch #ttm_on_fit_end to all plugins that export it.
     * @param[in] metrics_json JSON object containing final training metrics.
     */
		void emit_fit_end(std::string_view metrics_json);

		/** @} */

	private:
		/** @brief Private default constructor — use create() instead. */
		PluginManager() = default;

		/** @brief Internal per-plugin state — defined in plugin_manager.cpp. */
		struct Plugin;

		/** @brief Loaded plugins, in load order. */
		std::vector<std::unique_ptr<Plugin>> plugins;

		/**
     * @brief Source registry: scheme string → non-owning source pointer.
     *
     * Sources are owned by their Plugin record; this map holds raw pointers
     * for O(1) scheme lookup.
     */
		std::unordered_map<std::string, IDatasetSource*> sourceRegistry;

		/**
     * @brief True if this instance owns a WAMR runtime reference.
     *
     * @details Used by the move constructor/assignment and destructor to ensure
     *          wasm_loader_destroy() is called exactly once per successful
     *          wasm_loader_init() call.
     */
		bool wamrRefOwned = false;

		/* -----------------------------------------------------------------
     * Static host API callbacks (ctx == PluginManager*)
     * -------------------------------------------------------------- */

		/// @private
		static ttm_error s_register_source(void* ctx, const char** schemes, const ttm_source_vtable* vt);
		/// @private
		static ttm_error
		s_register_transform(void* ctx, const char* name, const char** aliases, const ttm_transform_vtable* vt);
		/// @private
		static ttm_error s_register_task(void* ctx, const char* name, const char** aliases, const ttm_task_vtable* vt);
		/// @private
		static ttm_error
		s_register_metric(void* ctx, const char* name, const char** aliases, const ttm_metric_vtable* vt);
		/// @private
		static void s_log(void* ctx, ttm_log_level level, const char* msg, uint32_t len);
		/// @private
		static void* s_alloc(void* ctx, uint32_t size);
		/// @private
		static void s_free(void* ctx, void* ptr);

		/**
     * @brief Construct a #ttm_host_api struct that points back to this manager.
     * @return Fully populated host API struct.
     */
		ttm_host_api make_host_api();

		/**
     * @brief Register a source object into the scheme registry.
     * @param[in] src    Source to register (ownership transferred).
     * @param[in] owner  Plugin record that produced this source.
     */
		void register_source_impl(std::unique_ptr<IDatasetSource> src, Plugin* owner);
	};

} // namespace ttm::plugins

#endif /* TTM_PLUGINS_PLUGIN_MANAGER_HPP */

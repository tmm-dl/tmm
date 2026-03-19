/**
 * @file abi.h
 * @brief TTM plugin ABI — the sole C contract between the host and any plugin.
 *
 * @details
 * Every plugin — whether compiled to WebAssembly or loaded as a native shared
 * library — must implement the three required exports declared at the bottom of
 * this file (#ttm_plugin_get_info, #ttm_plugin_init, #ttm_plugin_teardown) and
 * may implement any of the optional lifecycle hooks.
 *
 * Design rules:
 * - Pure C; no C++ constructs anywhere in this header.
 * - The POSIX-reserved `_t` suffix is intentionally omitted from all type names.
 * - Strings are (const char*, uint32_t len) pairs; not required to be NUL-terminated.
 * - Plugin-allocated memory returned via out-parameters must be freed by the host
 *   through #ttm_host_api::free.
 * - Object identity is represented as an opaque #ttm_handle (int64).
 *
 * @see ttm::plugins::IDatasetSource  C++ wrapper around #ttm_source_vtable
 * @see ttm::plugins::PluginManager   Host-side manager that loads plugins
 */

#ifndef TTM_PLUGINS_ABI_H
#define TTM_PLUGINS_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup abi_version Versioning
 * @{
 */

/** Current ABI version.  #ttm_plugin_info::abiVersion must equal this value. */
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) -- must be a macro: this header is included by C
// plugins where constexpr is unavailable
#define TTM_ABI_VERSION 1

/** @} */

/* =========================================================================
 * @defgroup abi_primitives Primitives
 * @{
 * ====================================================================== */

/**
 * @brief Opaque object reference used as "self" for vtable methods.
 * @details Obtained from a vtable's open/create function and passed back to
 *          every subsequent method call.  @see #TTM_INVALID_HANDLE
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header; `using` is a C++ construct
typedef int64_t ttm_handle;

/** Sentinel value returned by open/create functions on failure. */
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) -- must be a macro for use in C initializers
#define TTM_INVALID_HANDLE ((ttm_handle) - 1)

/**
 * @brief Status codes returned by most ABI functions.
 */
// NOLINTNEXTLINE(modernize-use-using,performance-enum-size) -- pure C header; enum base type must
// be int for C ABI compatibility (C does not support typed enums)
typedef enum {
	TTM_OK = 0,              /**< Success.                              */
	TTM_EOF = 1,             /**< End of stream / end of data.          */
	TTM_ERR_ARGS = 2,        /**< Invalid arguments.                    */
	TTM_ERR_NOT_FOUND = 3,   /**< Requested resource not found.         */
	TTM_ERR_IO = 4,          /**< I/O error.                            */
	TTM_ERR_OOM = 5,         /**< Out of memory.                        */
	TTM_ERR_UNSUPPORTED = 6, /**< Operation not supported.              */
} ttm_error;

/**
 * @brief Log severity levels passed to #ttm_host_api::log.
 */
// NOLINTNEXTLINE(modernize-use-using,performance-enum-size) -- same as ttm_error above
typedef enum {
	TTM_LOG_TRACE = 0,
	TTM_LOG_DEBUG = 1,
	TTM_LOG_INFO = 2,
	TTM_LOG_WARN = 3,
	TTM_LOG_ERROR = 4,
} ttm_log_level;

/** @} */

/* =========================================================================
 * @defgroup abi_vtables Vtables — COM-style OO in C
 * @{
 *
 * Each vtable describes a family of objects.  A concrete instance is
 * represented by a #ttm_handle obtained from the vtable's open/create
 * function and passed back as the first argument to every subsequent method.
 * ====================================================================== */

/**
 * @brief Vtable for byte-stream data sources.
 *
 * @details
 * A plugin registers one instance of this struct together with the URI schemes
 * it handles (e.g. `{"gh:", "github:", NULL}`).  The host calls open() with a
 * full URI; subsequent read()/seek()/close() calls operate on the returned
 * handle.
 *
 * @see ttm::plugins::IDatasetSource  C++ interface that wraps this vtable
 * @see ttm::plugins::IByteReader     C++ reader returned by IDatasetSource::open
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_source_vtable {
	/**
     * @brief Open a URI and return a handle to the byte stream.
     * @param uri      URI string (not NUL-terminated).
     * @param uri_len  Length of `uri` in bytes.
     * @param err_buf  Buffer to fill with a human-readable error message on failure.
     * @param err_cap  Capacity of `err_buf` in bytes.
     * @return A valid handle, or #TTM_INVALID_HANDLE on failure.
     */
	ttm_handle (*open)(const char* uri, uint32_t uri_len, char* err_buf, uint32_t err_cap);

	/**
     * @brief Read up to `len` bytes into `buf`.
     * @return Number of bytes read (>0), 0 at end-of-stream, or -1 on error.
     */
	int32_t (*read)(ttm_handle h, void* buf, int32_t len);

	/**
     * @brief Seek within the stream.
     * @param whence  Matches POSIX: 0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END.
     * @return New byte offset from the start, or -1 if unseekable / error.
     */
	int64_t (*seek)(ttm_handle h, int64_t offset, int32_t whence);

	/**
     * @brief Close the stream and release all resources for this handle.
     */
	void (*close)(ttm_handle h);
} ttm_source_vtable;

/**
 * @brief Vtable for Arrow IPC record-batch transformations.
 *
 * @details
 * Input and output are serialised Arrow IPC RecordBatch buffers.
 * The host owns `in_ipc`; the plugin allocates `out_ipc` and the host frees
 * it via #ttm_host_api::free.
 *
 * @see ttm::plugins::ITransform  C++ interface that wraps this vtable
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_transform_vtable {
	/**
     * @brief Create a transform instance.
     * @param config_json  Plugin-specific JSON configuration (not NUL-terminated).
     * @param config_len   Length of `config_json` in bytes.
     * @return Handle to the new instance, or #TTM_INVALID_HANDLE on failure.
     */
	ttm_handle (*create)(const char* config_json, uint32_t config_len);

	/**
     * @brief Apply the transform to a record batch.
     * @param h         Handle returned by create().
     * @param in_ipc    Serialised input Arrow IPC RecordBatch (host-owned).
     * @param in_len    Length of `in_ipc` in bytes.
     * @param out_ipc   Set to a plugin-allocated buffer containing the output batch.
     *                  The host will free this via #ttm_host_api::free.
     * @param out_len   Set to the length of `*out_ipc` in bytes.
     * @return #TTM_OK on success, otherwise an error code.
     */
	ttm_error (*apply)(ttm_handle h, const void* in_ipc, uint32_t in_len, void** out_ipc, uint32_t* out_len);

	/**
     * @brief Destroy a transform instance and release its resources.
     */
	void (*destroy)(ttm_handle h);
} ttm_transform_vtable;

/**
 * @brief Vtable for ML task types (text classification, NER, …).
 * @details
 * Task types correspond to the "task_categories" / "tasks" fields in
 * HuggingFace Dataset Cards.  A task may be registered under multiple
 * aliases (e.g. "text-classification" and "text-clf").
 * @see ttm::plugins::ITask
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_task_vtable {
	/** Canonical task name, e.g. "text-classification" (NUL-terminated). */
	const char* (*name)(void);

	/** NULL-terminated array of alias strings, e.g. {"text-clf","tc",NULL}.
	 *  May return NULL if there are no aliases. */
	const char** (*aliases)(void);

	/** NULL-terminated array of input feature names expected from the dataset
	 *  schema, e.g. {"text", NULL} for single-sentence classification. */
	const char** (*input_features)(void);

	/** Label feature name expected from the dataset schema, e.g. "label". */
	const char* (*label_feature)(void);

	/** NULL-terminated array of default evaluation metric names,
	 *  e.g. {"accuracy", "f1", NULL}. */
	const char** (*default_metrics)(void);
} ttm_task_vtable;

/**
 * @brief Vtable for evaluation metrics.
 * @details Full interface is TBD.
 * @see ttm::plugins::IMetric
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_metric_vtable {
	void* reserved; /**< Reserved — do not use. */
} ttm_metric_vtable;

/** @} */

/* =========================================================================
 * @defgroup abi_host_api Host API
 * @{
 *
 * The host builds a #ttm_host_api and passes a pointer to the plugin's
 * #ttm_plugin_init entry-point.  Plugins use these callbacks to register
 * extensions and access host services.
 * ====================================================================== */

/**
 * @brief Services and registration functions provided to plugins by the host.
 *
 * @details
 * A plugin receives a `const ttm_host_api*` in #ttm_plugin_init and must not
 * retain this pointer after #ttm_plugin_teardown returns.
 *
 * @see ttm::plugins::PluginManager::load  Where this struct is constructed
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_host_api {
	/**
     * @brief Register a byte-stream source for a set of URI schemes.
     * @param ctx      Opaque host token (pass back as-is).
     * @param schemes  NULL-terminated array of NUL-terminated scheme strings,
     *                 e.g. `{"gh:", "github:", NULL}`.
     * @param vt       Vtable implementing the source.
     * @return #TTM_OK on success.
     * @see ttm::plugins::IDatasetSource
     */
	ttm_error (*register_source)(void* ctx, const char** schemes, const ttm_source_vtable* vt);

	/**
     * @brief Register a record-batch transform.
     * @param ctx      Opaque host token.
     * @param name     Canonical NUL-terminated name.
     * @param aliases  NULL-terminated array of additional NUL-terminated names
     *                 (may be NULL).
     * @param vt       Vtable implementing the transform.
     * @return #TTM_OK on success.
     * @see ttm::plugins::ITransform
     */
	ttm_error (*register_transform)(void* ctx, const char* name, const char** aliases, const ttm_transform_vtable* vt);

	/**
     * @brief Register an ML task type.
     * @param ctx      Opaque host token.
     * @param name     Canonical NUL-terminated name (e.g. "text-classification").
     * @param aliases  NULL-terminated array of additional names (may be NULL).
     * @param vt       Vtable implementing the task.
     * @return #TTM_OK on success.
     * @see ttm::plugins::ITask
     */
	ttm_error (*register_task)(void* ctx, const char* name, const char** aliases, const ttm_task_vtable* vt);

	/**
     * @brief Register an evaluation metric.
     * @param ctx      Opaque host token.
     * @param name     Canonical NUL-terminated name (e.g. "f1").
     * @param aliases  NULL-terminated array of additional names (may be NULL).
     * @param vt       Vtable implementing the metric.
     * @return #TTM_OK on success.
     * @see ttm::plugins::IMetric
     */
	ttm_error (*register_metric)(void* ctx, const char* name, const char** aliases, const ttm_metric_vtable* vt);

	/**
     * @brief Emit a log message to the host logger.
     * @param ctx    Opaque host token.
     * @param level  Severity level.
     * @param msg    Message bytes (not required to be NUL-terminated).
     * @param len    Length of `msg` in bytes.
     */
	void (*log)(void* ctx, ttm_log_level level, const char* msg, uint32_t len);

	/**
     * @brief Allocate `size` bytes in the plugin's address space.
     * @details For WASM plugins this allocates within the module's linear memory
     *          so that returned pointers are valid from the plugin side.
     * @return Pointer to the allocated block, or NULL on failure.
     */
	void* (*alloc)(void* ctx, uint32_t size);

	/**
     * @brief Free memory previously obtained via #ttm_host_api::alloc.
     */
	void (*free)(void* ctx, void* ptr);

	/** @brief Opaque token passed back as the first argument to every callback. */
	void* ctx;
} ttm_host_api;

/** @} */

/* =========================================================================
 * @defgroup abi_plugin_metadata Plugin metadata
 * @{
 * ====================================================================== */

/**
 * @brief Static metadata returned by #ttm_plugin_get_info.
 * @see ttm_plugin_get_info
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_plugin_info {
	uint32_t    abiVersion;  /**< Must equal #TTM_ABI_VERSION.               */
	const char* name;        /**< Human-readable plugin name (NUL-terminated). */
	const char* version;     /**< SemVer string, e.g. "1.0.0" (NUL-terminated). */
	const char* description; /**< One-line description (NUL-terminated; may be NULL). */
} ttm_plugin_info;

/** @} */

/* =========================================================================
 * @defgroup abi_required_exports Required plugin exports
 * @{
 *
 * Every plugin must export these three symbols.  The host looks them up by
 * name immediately after loading the module.
 * ====================================================================== */

/**
 * @brief Return static plugin metadata.
 * @details Called before #ttm_plugin_init; must be safe to call at any time
 *          and must not allocate or perform I/O.
 * @return Pointer to a static #ttm_plugin_info; never NULL.
 * @see ttm_plugin_info
 */
ttm_plugin_info* ttm_plugin_get_info(void);

/**
 * @brief Initialise the plugin and register all extension points.
 * @details
 * The host calls this once per load after verifying the ABI version.
 * The plugin must call the appropriate `host->register_*()` functions here;
 * registration after this call returns is not supported.
 *
 * @param host         Host API callbacks and context.  Valid until
 *                     #ttm_plugin_teardown returns.
 * @param config_json  Plugin-specific JSON configuration (not NUL-terminated).
 *                     The host passes `"{}"` when no configuration is supplied.
 * @param config_len   Length of `config_json` in bytes.
 * @return #TTM_OK on success; any other value causes the host to unload the plugin.
 * @see ttm_host_api
 */
ttm_error ttm_plugin_init(const ttm_host_api* host, const char* config_json, uint32_t config_len);

/**
 * @brief Tear down the plugin.
 * @details Called once on unload.  All handles issued by this plugin are
 *          invalid after this function returns.
 * @see ttm_plugin_init
 */
void ttm_plugin_teardown(void);

/** @} */

/* =========================================================================
 * @defgroup abi_lifecycle_hooks Optional lifecycle hooks
 * @{
 *
 * The host looks up these symbols by name after #ttm_plugin_init returns.
 * Missing symbols are silently ignored — plugins implement only the hooks
 * they need.  All hooks are called on the same thread.
 * ====================================================================== */

/**
 * @brief Called once before the fit loop begins.
 * @param ctx_json  JSON object carrying run metadata (e.g. hyperparameters).
 * @param len       Length of `ctx_json` in bytes.
 */
void ttm_on_fit_begin(const char* ctx_json, uint32_t len);

/**
 * @brief Called at the start of each epoch.
 * @param epoch        0-based current epoch index.
 * @param total_epochs Total number of epochs planned.
 */
void ttm_on_epoch_begin(uint32_t epoch, uint32_t total_epochs);

/**
 * @brief Called at the start of each training batch.
 * @param batch         0-based current batch index within the epoch.
 * @param total_batches Total batches in the epoch.
 */
void ttm_on_batch_begin(uint32_t batch, uint32_t total_batches);

/**
 * @brief Called immediately after the loss is computed.
 * @details The plugin may return a modified loss value.  Returning the input
 *          unchanged is the default behaviour.  The host chains this call
 *          across all plugins in load order.
 * @param loss  The loss value computed by the training loop.
 * @return Possibly modified loss value passed to subsequent plugins.
 */
float ttm_on_loss_computed(float loss);

/**
 * @brief Called after a training batch completes.
 * @param batch        0-based batch index.
 * @param loss         Final (possibly modified) loss for this batch.
 * @param metrics_json JSON object containing live scalar metrics.
 * @param len          Length of `metrics_json` in bytes.
 */
void ttm_on_batch_end(uint32_t batch, float loss, const char* metrics_json, uint32_t len);

/**
 * @brief Called at the end of each epoch.
 * @details All plugins are invoked regardless of earlier return values so that
 *          every plugin sees the epoch end.  The host ORs return values across
 *          all plugins to determine whether to stop early.
 * @param epoch        0-based epoch index.
 * @param metrics_json JSON object containing epoch-level metrics.
 * @param len          Length of `metrics_json` in bytes.
 * @return Non-zero to request early stopping.
 */
int32_t ttm_on_epoch_end(uint32_t epoch, const char* metrics_json, uint32_t len);

/**
 * @brief Called after each validation pass.
 * @param metrics_json JSON object containing validation metrics.
 * @param len          Length of `metrics_json` in bytes.
 */
void ttm_on_validation_end(const char* metrics_json, uint32_t len);

/**
 * @brief Called once when the fit loop finishes (normally or via early stop).
 * @param metrics_json JSON object containing final metrics.
 * @param len          Length of `metrics_json` in bytes.
 */
void ttm_on_fit_end(const char* metrics_json, uint32_t len);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TTM_PLUGINS_ABI_H */

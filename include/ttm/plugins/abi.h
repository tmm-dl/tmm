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

/* DLPack — standard tensor interchange format used by the model loader ABI.
 * dlpack.h is a self-contained, pure-C header. */
#include <dlpack/dlpack.h>

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
 * @defgroup abi_model_loader Model loader vtable
 * @{
 *
 * Plugins implement ttm_model_loader_vtable and register it via
 * ttm_host_api::register_model_loader.  The host calls probe() to decide
 * which registered loader owns a given file, then load() to instantiate
 * the model.
 * ====================================================================== */

/**
 * @brief Per-parameter layout descriptor returned by describe_params().
 *
 * @details
 * All pointer fields are valid until the model is destroyed.  The host must
 * not free them.
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_param_desc_t {
	const char*    name;         /**< Parameter name (NUL-terminated).               */
	int32_t        ndim;         /**< Number of tensor dimensions.                   */
	int32_t        dtype_code;   /**< DLDataTypeCode: 0=int, 1=uint, 2=float, …     */
	int32_t        dtype_bits;   /**< Bit-width of the element type (e.g. 32).       */
	const int64_t* shape;        /**< Shape array of length ndim (plugin-owned).     */
	int32_t        trainable;    /**< Non-zero if gradients should be tracked.       */
} ttm_param_desc_t;

/**
 * @brief Model metadata broadcast to plugins via ttm_on_model_loaded.
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_model_info_t {
	const char* name;               /**< Human-readable model name (NUL-terminated).   */
	const char* arch;               /**< Architecture tag, e.g. "GPT2" (NUL-term.).   */
	uint64_t    num_parameters;     /**< Total parameter count.                        */
	uint64_t    num_trainable;      /**< Count of trainable parameters.                */
	uint64_t    bytes_on_device;    /**< Memory footprint (params + grads), bytes.     */
	const char* input_schema_json;  /**< Arrow JSON schema for expected inputs.        */
	int32_t     device_type;        /**< DLDeviceType of the device the model lives on. */
	int32_t     device_id;          /**< Device index (e.g. GPU ordinal).              */
} ttm_model_info_t;

/**
 * @brief Vtable for plugin-provided model loaders.
 *
 * @details
 * A plugin registers one instance of this struct via
 * #ttm_host_api::register_model_loader.  The host uses probe() to find the
 * right loader for a given file, then calls the remaining methods to set up,
 * train, and tear down the model.
 *
 * All methods operate on an opaque #ttm_handle returned by load().  The host
 * owns the DLTensor buffers passed to bind_params(); the plugin must not free
 * them.
 *
 * @see ttm::plugins::IModelLoader  C++ wrapper around this vtable
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_model_loader_vtable {
	/**
	 * @brief Return non-zero if this loader can handle the given path.
	 * @param path    File path (not NUL-terminated).
	 * @param len     Length of `path` in bytes.
	 */
	int32_t (*probe)(const char* path, uint32_t len);

	/**
	 * @brief Load a model from disk and return a handle.
	 * @param path      Model file path (not NUL-terminated).
	 * @param path_len  Length of `path` in bytes.
	 * @param cfg       Plugin-specific JSON config (not NUL-terminated).
	 * @param cfg_len   Length of `cfg` in bytes.
	 * @param err       Buffer for a human-readable error message on failure.
	 * @param err_cap   Capacity of `err` in bytes.
	 * @return A valid handle, or #TTM_INVALID_HANDLE on failure.
	 */
	ttm_handle (*load)(const char* path, uint32_t path_len,
	                   const char* cfg,  uint32_t cfg_len,
	                   char* err, uint32_t err_cap);

	/**
	 * @brief Return static metadata for a loaded model.
	 * @details All pointer fields in the returned struct are valid until
	 *          destroy() is called on the same handle.
	 */
	ttm_model_info_t (*get_info)(ttm_handle h);

	/**
	 * @brief Describe the parameter layout.
	 * @details The returned array is valid until the next call to this method
	 *          or until destroy().
	 * @param[out] out_descs  Set to a plugin-owned array of descriptors.
	 * @param[out] out_count  Set to the number of entries in `*out_descs`.
	 */
	ttm_error (*describe_params)(ttm_handle h,
	                             const ttm_param_desc_t** out_descs,
	                             uint32_t* out_count);

	/**
	 * @brief Bind host-allocated parameter and gradient buffers to the model.
	 * @details Called once after load(), before any step()/infer() calls.
	 *          The host retains ownership of all DLTensors.
	 */
	ttm_error (*bind_params)(ttm_handle h,
	                         const DLTensor* params, uint32_t param_count,
	                         const DLTensor* grads,  uint32_t grad_count);

	/**
	 * @brief Initialise parameter values.
	 * @param method  Either "random" or "checkpoint:<path>" (not NUL-terminated).
	 * @param len     Length of `method` in bytes.
	 */
	ttm_error (*init_params)(ttm_handle h, const char* method, uint32_t len);

	/**
	 * @brief Forward + backward pass; writes the scalar loss to `out_loss`.
	 * @param inputs   Array of input DLTensors (host-owned).
	 * @param n        Number of input tensors.
	 * @param out_loss Set to the scalar loss for this batch.
	 */
	ttm_error (*step)(ttm_handle h,
	                  const DLTensor* inputs, uint32_t n,
	                  float* out_loss);

	/**
	 * @brief Forward-only pass; fills caller-provided output tensors.
	 * @param inputs      Array of input DLTensors (host-owned).
	 * @param in_count    Number of input tensors.
	 * @param outputs     Array of output DLTensors to fill (host-owned).
	 * @param out_count   On entry, capacity of `outputs`; on exit, filled count.
	 */
	ttm_error (*infer)(ttm_handle h,
	                   const DLTensor* inputs,  uint32_t in_count,
	                   DLTensor*       outputs, uint32_t* out_count);

	/** @brief Zero all gradient buffers. */
	ttm_error (*zero_grad)(ttm_handle h);

	/** @brief Destroy the model and release all plugin-side resources. */
	void (*destroy)(ttm_handle h);
} ttm_model_loader_vtable;

/** @} */

/* =========================================================================
 * @defgroup abi_scheduler LR scheduler vtable
 * @{
 *
 * Plugins implement ttm_scheduler_vtable and register it by name via
 * ttm_host_api::register_scheduler.  The host calls create() once before
 * the fit loop and step() after each optimizer step.
 * ====================================================================== */

/**
 * @brief Vtable for learning-rate schedulers.
 *
 * @details
 * A plugin registers one or more schedulers by calling
 * `host->register_scheduler(ctx, "cosine_warmup", &vt)`.  The host looks
 * up the scheduler named in the training config and uses this vtable to
 * drive the LR over the course of training.
 *
 * @see ttm::trainer::ILRScheduler  C++ interface backed by this vtable
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_scheduler_vtable {
	/**
	 * @brief Create a scheduler instance.
	 * @param base_lr   Initial (peak) learning rate from the optimizer config.
	 * @param cfg       JSON object with scheduler-specific fields
	 *                  (e.g. warmup_steps, min_lr, total_steps).
	 * @param cfg_len   Length of `cfg` in bytes.
	 * @return Opaque handle, or #TTM_INVALID_HANDLE on failure.
	 */
	ttm_handle (*create)(float base_lr, const char* cfg, uint32_t cfg_len);

	/**
	 * @brief Compute and return the LR for the given global optimizer step.
	 * @param h           Handle returned by create().
	 * @param global_step Monotonically increasing optimizer step index (0-based).
	 * @return Learning rate to use for the next optimizer update.
	 */
	float (*step)(ttm_handle h, int64_t global_step);

	/**
	 * @brief Destroy the scheduler instance and release resources.
	 */
	void (*destroy)(ttm_handle h);
} ttm_scheduler_vtable;

/** @} */

/* =========================================================================
 * @defgroup abi_optimizer Optimizer vtable
 * @{
 *
 * Plugins implement ttm_optimizer_vtable and register it by name via
 * ttm_host_api::register_optimizer.  The host calls create() once after the
 * model is loaded, passing the model handle so that same-plugin optimizers
 * can access model internals directly (e.g. the Python plugin calling
 * model.parameters()).  For cross-plugin usage, host-allocated param/grad
 * DLTensors are passed instead.
 * ====================================================================== */

/**
 * @brief Vtable for gradient-based parameter optimizers.
 *
 * @details
 * A plugin registers one or more optimizers by calling
 * `host->register_optimizer(ctx, "adamw", &vt)`.  The training config
 * selects an optimizer by this name.
 *
 * ### Cross-plugin compatibility
 * When model and optimizer live in the same plugin (e.g. both PyTorch),
 * @p model_h lets the optimizer access the model's internal state directly.
 * When they live in different plugins, the host passes the model's param and
 * grad DLTensors so the optimizer can work on them generically.
 *
 * ### LR scheduler integration
 * The host calls set_lr() after each scheduler step so the optimizer uses
 * the updated learning rate on the next step() call.
 *
 * @see ttm::trainer::IOptimizer  C++ interface backed by this vtable
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_optimizer_vtable {
	/**
     * @brief Create an optimizer instance.
     *
     * @param model_h     Handle from model_loader.load() for the model being
     *                    optimized.  Same-plugin optimizers may use this to
     *                    access the underlying model object (e.g. to call
     *                    model.parameters()).  Pass #TTM_INVALID_HANDLE if the
     *                    model is external to this plugin.
     * @param params      Host-allocated param DLTensors (may be nullptr when
     *                    the model manages its own parameter memory).
     * @param param_count Number of elements in @p params (and @p grads).
     * @param grads       Host-allocated gradient DLTensors (same count as
     *                    @p params; may be nullptr).
     * @param cfg         JSON object with optimizer hyperparameters.
     *                    Standard fields: lr, weight_decay, beta1, beta2, eps,
     *                    amsgrad (0/1), device (e.g. "cpu", "cuda", "cuda:1").
     * @param cfg_len     Length of @p cfg in bytes.
     * @param err         Buffer for a human-readable error message on failure.
     * @param err_cap     Capacity of @p err in bytes.
     * @return Opaque handle, or #TTM_INVALID_HANDLE on failure.
     */
	ttm_handle (*create)(ttm_handle      model_h,
	                     const DLTensor* params, uint32_t param_count,
	                     const DLTensor* grads,
	                     const char* cfg, uint32_t cfg_len,
	                     char* err, uint32_t err_cap);

	/**
     * @brief Apply one optimizer step using the gradients currently stored in
     *        the model.
     */
	ttm_error (*step)(ttm_handle h);

	/** @brief Zero all gradient accumulators. */
	ttm_error (*zero_grad)(ttm_handle h);

	/** @brief Return the current learning rate for the first param group. */
	float (*get_lr)(ttm_handle h);

	/**
     * @brief Update the learning rate for all param groups.
     * @details Called by the host after each LR scheduler step.
     */
	void (*set_lr)(ttm_handle h, float lr);

	/** @brief Destroy the optimizer and release all plugin-side resources. */
	void (*destroy)(ttm_handle h);
} ttm_optimizer_vtable;

/** @} */

/* =========================================================================
 * @defgroup abi_callback Trainer callback vtable
 * @{
 *
 * Plugins implement ttm_trainer_callback_vtable and register it by name via
 * ttm_host_api::register_callback.  The host creates instances from the
 * training config's callbacks[] list.
 * ====================================================================== */

/**
 * @brief Vtable for trainer lifecycle callbacks.
 *
 * @details
 * A plugin registers one or more callbacks by calling
 * `host->register_callback(ctx, "early_stopping", &vt)`.  The training
 * config selects callbacks by name (optionally qualified as "plugin::name").
 */
// NOLINTNEXTLINE(modernize-use-using) -- pure C header
typedef struct ttm_trainer_callback_vtable {
	/**
	 * @brief Create a callback instance.
	 * @param config_json  JSON with callback-specific settings (not NUL-terminated).
	 * @param config_len   Length of config_json in bytes.
	 * @return Opaque handle, or #TTM_INVALID_HANDLE on failure.
	 */
	ttm_handle (*create)(const char* config_json, uint32_t config_len);

	/**
	 * @brief Called once before the fit loop begins.
	 * @param h            Handle returned by create().
	 * @param metrics_json JSON object with initial metrics (may be empty).
	 * @param len          Length of metrics_json in bytes.
	 */
	void (*on_fit_begin)(ttm_handle h, const char* metrics_json, uint32_t len);

	/**
	 * @brief Called at the start of each epoch.
	 * @param h      Handle returned by create().
	 * @param epoch  0-based epoch index.
	 * @param total  Total number of planned epochs.
	 */
	void (*on_epoch_begin)(ttm_handle h, int64_t epoch, int64_t total);

	/**
	 * @brief Called at the end of each epoch.
	 * @param h            Handle returned by create().
	 * @param epoch        0-based epoch index.
	 * @param metrics_json JSON object with epoch-level metrics (e.g. val_loss).
	 * @param len          Length of metrics_json in bytes.
	 * @return Non-zero to request early stopping.
	 */
	int32_t (*on_epoch_end)(ttm_handle h, int64_t epoch,
	                        const char* metrics_json, uint32_t len);

	/**
	 * @brief Called once after the fit loop ends.
	 * @param h            Handle returned by create().
	 * @param metrics_json JSON object with final metrics.
	 * @param len          Length of metrics_json in bytes.
	 */
	void (*on_fit_end)(ttm_handle h, const char* metrics_json, uint32_t len);

	/** @brief Destroy the callback instance and release resources. */
	void (*destroy)(ttm_handle h);
} ttm_trainer_callback_vtable;

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
     * @brief Log a named scalar metric (PyTorch Lightning–style).
     * @details
     * Plugins and training code call this to record a scalar value at a given
     * global optimiser step.  The host broadcasts the metric to every other
     * loaded plugin via #ttm_on_metric so that UI or logging plugins can
     * display it without knowing the source.
     *
     * @param ctx      Opaque host token.
     * @param key      Metric name (not required to be NUL-terminated).
     * @param key_len  Length of `key` in bytes.
     * @param value    Scalar value (NaN / Inf are valid; consumers should handle them).
     * @param step     Global training step at which the value was recorded.
     */
	void (*log_metric)(void* ctx, const char* key, uint32_t key_len, float value, int32_t step);

	/**
     * @brief Query the host terminal dimensions.
     * @details
     * WASM plugins cannot call ioctl directly; this callback lets them query
     * the real terminal size from the host so they can size their rendering
     * buffers accordingly.
     *
     * @param ctx        Opaque host token.
     * @param out_width  Set to the number of terminal columns (default 80).
     * @param out_height Set to the number of terminal rows    (default 24).
     */
	void (*terminal_size)(void* ctx, uint32_t* out_width, uint32_t* out_height);

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

	/**
     * @brief Register a model loader.
     * @details Called by a plugin during #ttm_plugin_init to register a
     *          vtable that the host will use to load models matching the
     *          files accepted by vt->probe().
     * @param ctx  Opaque host token.
     * @param vt   Model loader vtable.  The pointer must remain valid for the
     *             lifetime of the plugin (i.e. until #ttm_plugin_teardown).
     * @return #TTM_OK on success.
     * @see ttm_model_loader_vtable
     */
	ttm_error (*register_model_loader)(void* ctx, const ttm_model_loader_vtable* vt);

	/**
     * @brief Broadcast model metadata to all loaded plugins.
     * @details Called by the host after a model is successfully loaded.
     *          Plugins that export #ttm_on_model_loaded will receive the JSON.
     * @param ctx   Opaque host token.
     * @param info  Model metadata to broadcast.
     * @see ttm_on_model_loaded
     */
	void (*notify_model_info)(void* ctx, const ttm_model_info_t* info);

	/**
     * @brief Register an LR scheduler under a given name.
     * @details Called by a plugin during #ttm_plugin_init.  The host stores a
     *          copy of the vtable struct and associates it with `name`.  The
     *          training config selects a scheduler by this name.
     * @param ctx   Opaque host token.
     * @param name  NUL-terminated scheduler name (e.g. "cosine_warmup").
     * @param vt    Scheduler vtable.  The function pointers must remain valid
     *              for the lifetime of the plugin.
     * @return #TTM_OK on success.
     * @see ttm_scheduler_vtable
     */
	ttm_error (*register_scheduler)(void* ctx, const char* name, const ttm_scheduler_vtable* vt);

	/**
     * @brief Register an optimizer under a given name.
     * @details Called by a plugin during #ttm_plugin_init.  The training config
     *          selects an optimizer by this name (e.g. "adamw").
     * @param ctx   Opaque host token.
     * @param name  NUL-terminated optimizer name (e.g. "adamw").
     * @param vt    Optimizer vtable.  The function pointers must remain valid
     *              for the lifetime of the plugin.
     * @return #TTM_OK on success.
     * @see ttm_optimizer_vtable
     */
	ttm_error (*register_optimizer)(void* ctx, const char* name, const ttm_optimizer_vtable* vt);

	/**
	 * @brief Register a trainer callback under a given name.
	 * @details Called by a plugin during #ttm_plugin_init.  The training config
	 *          selects callbacks by name (e.g. "early_stopping").
	 * @param ctx   Opaque host token.
	 * @param name  NUL-terminated callback name (e.g. "early_stopping").
	 * @param vt    Callback vtable.  The function pointers must remain valid
	 *              for the lifetime of the plugin.
	 * @return #TTM_OK on success.
	 * @see ttm_trainer_callback_vtable
	 */
	ttm_error (*register_callback)(void* ctx, const char* name, const ttm_trainer_callback_vtable* vt);

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

/**
 * @brief Called when the host emits a log message.
 * @details
 * Plugins that export this symbol receive every log message emitted by the
 * host (trainer progress, warnings, errors, etc.).  This allows UI plugins
 * to capture and display logs without redirecting file descriptors.
 *
 * The host skips the default stderr fallback when at least one loaded plugin
 * exports this symbol, so the plugin is responsible for surfacing the message.
 *
 * @param level  Severity — one of the #ttm_log_level values.
 * @param msg    Message bytes (not NUL-terminated).
 * @param len    Length of `msg` in bytes.
 */
void ttm_on_log(uint32_t level, const char* msg, uint32_t len);

/**
 * @brief Called when a named scalar metric is logged via #ttm_host_api::log_metric.
 * @details
 * Plugins that export this symbol receive every metric logged by the host or
 * by other plugins.  The originating plugin does not receive its own metric
 * back to avoid loops.
 *
 * @param key      Metric name (not NUL-terminated).
 * @param key_len  Length of `key` in bytes.
 * @param value    Scalar value.
 * @param step     Global training step.
 */
void ttm_on_metric(const char* key, uint32_t key_len, float value, int32_t step);

/**
 * @brief Called after a model is successfully loaded.
 * @details
 * The host serialises the #ttm_model_info_t into a JSON object and broadcasts
 * it to every loaded plugin that exports this symbol.  UI plugins (console-ui,
 * inspector) use this to display model metadata such as the architecture name,
 * parameter count, and memory footprint.
 *
 * @param info_json  JSON-serialised #ttm_model_info_t (not NUL-terminated).
 * @param len        Length of `info_json` in bytes.
 */
void ttm_on_model_loaded(const char* info_json, uint32_t len);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TTM_PLUGINS_ABI_H */

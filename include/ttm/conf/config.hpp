/**
 * @file config.hpp
 * @brief Typed training-configuration structures.
 *
 * @details
 * These plain-data structs represent a fully-resolved training configuration.
 * They are populated by @ref ttm::conf::load_config after YAML parsing, deep
 * merge, `--set` override application, and `${ENV_VAR}` interpolation.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ttm::conf {

	/* =========================================================================
	 * Sub-configs
	 * ====================================================================== */

	struct DatasetConfig {
		std::string uri;              ///< Dataset URI (e.g. `hf:thagen/SCITE`, `file:/data`)
		std::string config_name;      ///< HF named config (e.g. `"causality detection"`)
		std::string split = "train";
		int64_t     batch_size            = 32;
		bool        shuffle               = true;
		int64_t     shuffle_buffer_size   = 10'000;
		int64_t     num_workers           = 4;
		int64_t     prefetch              = 2;
	};

	struct ValidationConfig {
		std::string uri;              ///< If empty, reuses `dataset.uri`
		std::string config_name;
		std::string split      = "validation";
		int64_t     batch_size = 32;
	};

	struct ModelConfig {
		std::string path;                        ///< Path to compiled TVM module or shared lib
		std::string function_name = "main";      ///< Entry-point function name
		std::string device        = "cpu";       ///< `cpu`, `cuda`, `metal`, …
		int32_t     device_id     = 0;
	};

	struct OptimizerConfig {
		std::string type = "adamw"; ///< `sgd` | `adam` | `adamw` | `rmsprop`
		float lr           = 1e-3f;
		float weight_decay = 1e-2f;
		float momentum     = 0.9f;   ///< SGD momentum
		float beta1        = 0.9f;   ///< Adam β₁
		float beta2        = 0.999f; ///< Adam β₂
		float eps          = 1e-8f;
		bool  nesterov     = false;  ///< Nesterov SGD
		bool  amsgrad      = false;  ///< AMSGrad variant of Adam
	};

	struct SchedulerConfig {
		/// `constant` | `step` | `linear` | `cosine` | `cosine_warmup`
		std::string type         = "cosine_warmup";
		int64_t     warmup_steps = 0;
		float       min_lr       = 0.0f; ///< Floor for cosine / linear decay
		int64_t     step_size    = 1;    ///< StepLR: decay every N steps
		float       gamma        = 0.1f; ///< StepLR: multiplicative decay
		int64_t     total_steps  = 0;    ///< Total optimizer steps (0 = no decay limit)
	};

	struct CheckpointConfig {
		std::string dir                  = "checkpoints";
		int32_t     save_every_n_epochs  = 1;
		int32_t     keep_top_k           = 3;
		std::string monitor              = "val_loss"; ///< Metric to track
		std::string monitor_mode         = "min";      ///< `min` | `max`
	};

	struct PluginEntry {
		std::string name;   ///< Logical name (e.g. `"core"`, `"python"`); resolved to a path at load time
		std::string path;   ///< Path to `.wasm` or `.so`/`.dylib`/`.dll`; takes priority over `name` if set
		std::string config; ///< Arbitrary JSON passed to `ttm_plugin_init`
	};

	/**
	 * @brief Configuration entry for a preprocessor plugin.
	 *
	 * @details
	 * Preprocessors are applied to each training batch in order before the batch
	 * is collated into DLTensors for the model.  Each entry maps to a registered
	 * ITransform (identified by `type`) and receives the JSON `config`.
	 *
	 * ### YAML example
	 * @code{.yaml}
	 * preprocessors:
	 *   - type: bpe-tokenize
	 *     config: '{"vocab": "vocab.json", "max_length": 512}'
	 *   - type: truncate
	 *     config: '{"max_length": 512}'
	 * @endcode
	 */
	struct PreprocessorEntry {
		std::string type;   ///< Registered transform name, e.g. `"bpe-tokenize"`.
		std::string config; ///< JSON passed to ITransform::create() (default: `"{}"`).
	};

	/**
	 * @brief Configuration entry for a plugin-provided trainer callback.
	 *
	 * @details
	 * Callbacks are loaded by name from the plugin registry.  The optional
	 * namespace qualifier selects the owning plugin explicitly:
	 *   - `type: early_stopping`       → first plugin that registered it wins
	 *   - `type: core::early_stopping` → always loaded from the core plugin
	 *
	 * The `config` field is passed verbatim as JSON to the callback's `create()`
	 * vtable function.
	 *
	 * ### YAML example
	 * @code{.yaml}
	 * callbacks:
	 *   - type: early_stopping
	 *     config: '{"monitor":"val_loss","patience":5,"mode":"min","min_delta":0.0}'
	 *   - type: core::checkpoint
	 *     config: '{"directory":"./checkpoints","every_n_epochs":1}'
	 * @endcode
	 */
	struct CallbackEntry {
		std::string type;            ///< Callback name, optionally qualified (e.g. `"early_stopping"`, `"core::checkpoint"`)
		std::string config = "{}";   ///< JSON passed to the callback's `create()` vtable function
	};

	/* =========================================================================
	 * Top-level config
	 * ====================================================================== */

	/**
	 * @brief Fully resolved training configuration.
	 *
	 * @details
	 * This struct is the result of merging one or more YAML files and applying
	 * CLI overrides.  All fields have sensible defaults so that a minimal config
	 * only needs to specify `dataset.uri` and `model.path`.
	 */
	struct TrainingConfig {
		std::string version = "1";

		DatasetConfig                   dataset;
		std::optional<ValidationConfig> validation;
		ModelConfig                     model;
		OptimizerConfig                 optimizer;
		SchedulerConfig                 scheduler;
		CheckpointConfig                checkpoint;
		std::vector<PluginEntry>        plugins;
		std::vector<PreprocessorEntry>  preprocessors;
		std::vector<CallbackEntry>      callbacks;

		/* --- training loop knobs --- */
		int64_t epochs                      = 10;
		int64_t gradient_accumulation_steps = 1;
		float   grad_clip_norm              = 0.0f; ///< 0 = disabled
		bool    fp16                        = false;
		int64_t seed                        = 42;
		std::string log_level               = "info";
	};

} // namespace ttm::conf

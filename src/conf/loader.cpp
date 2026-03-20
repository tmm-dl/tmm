/**
 * @file loader.cpp
 * @brief Config file loading, YAML deep-merge, override application,
 *        environment interpolation, and node → TrainingConfig conversion.
 */

#include <ttm/conf/loader.hpp>
#include <ttm/compat/format.hpp>

#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ttm::conf {

	namespace {

		/* =====================================================================
		 * YAML utilities
		 * ================================================================== */

		/**
		 * @brief Deep-merge `overlay` into `base` (in-place).
		 *
		 * Maps are merged recursively; scalars and sequences are replaced by the
		 * overlay value (right-wins semantics matching Helm chart merging).
		 */
		void deep_merge(YAML::Node base, const YAML::Node& overlay) {
			if (!overlay.IsMap() || !base.IsMap()) return;
			for (const auto& kv : overlay) {
				const std::string key = kv.first.as<std::string>();
				if (base[key] && base[key].IsMap() && kv.second.IsMap()) {
					deep_merge(base[key], kv.second);
				} else {
					base[key] = kv.second;
				}
			}
		}

		/**
		 * @brief Walk all scalar string nodes and expand `${VAR_NAME}` env references.
		 */
		void interpolate_env(YAML::Node node) {
			if (node.IsScalar()) {
				std::string val = node.as<std::string>();
				std::string result;
				result.reserve(val.size());
				std::size_t pos = 0;
				while (pos < val.size()) {
					const std::size_t start = val.find("${", pos);
					if (start == std::string::npos) {
						result.append(val, pos);
						break;
					}
					result.append(val, pos, start - pos);
					const std::size_t end = val.find('}', start + 2);
					if (end == std::string::npos) {
						result.append(val, start);
						break;
					}
					const std::string varName = val.substr(start + 2, end - start - 2);
					// NOLINTNEXTLINE(concurrency-mt-unsafe)
					const char* envVal = std::getenv(varName.c_str());
					if (envVal != nullptr) result += envVal;
					pos = end + 1;
				}
				if (result != val) node = result;
			} else if (node.IsMap()) {
				for (auto kv : node) interpolate_env(kv.second);
			} else if (node.IsSequence()) {
				for (auto item : node) interpolate_env(item);
			}
		}

		/**
		 * @brief Recursively navigate `keys[idx…]` into `node` and set the leaf.
		 *
		 * yaml-cpp's `operator=` copies data rather than rebinding the internal
		 * node pointer, so intermediate reassignment (`cur = cur[k]`) would detach
		 * from the tree.  Passing `node[keys[idx]]` directly as the next recursive
		 * argument uses the copy-constructor, which preserves the shared reference
		 * into the document tree.
		 */
		static void set_nested(
			YAML::Node              node,
			const std::vector<std::string>& keys,
			std::size_t             idx,
			const std::string&      value
		) {
			if (idx + 1 == keys.size()) {
				node[keys[idx]] = value;
				return;
			}
			set_nested(node[keys[idx]], keys, idx + 1, value);
		}

		/**
		 * @brief Apply a single `"key.path=value"` override to the YAML root node.
		 *
		 * Intermediate mapping nodes are created automatically.  The leaf is always
		 * written as a YAML scalar string; yaml-cpp converts to the target type when
		 * read via `.as<T>()`.
		 */
		std::expected<void, std::string>
		apply_override(YAML::Node root, std::string_view kv) {
			const auto eq = kv.find('=');
			if (eq == std::string_view::npos) {
				return std::unexpected(
					std::format("load_config: invalid --set value '{}' (expected key=value)", kv)
				);
			}
			const std::string key_path{kv.substr(0, eq)};
			const std::string value{kv.substr(eq + 1)};

			std::vector<std::string> keys;
			std::istringstream       ss{key_path};
			std::string              part;
			while (std::getline(ss, part, '.')) {
				if (!part.empty()) keys.push_back(std::move(part));
			}
			if (keys.empty()) {
				return std::unexpected(
					std::format("load_config: empty key in --set '{}'", kv)
				);
			}

			set_nested(root, keys, 0, value);
			return {};
		}

		/* =====================================================================
		 * Typed-field extraction helpers
		 * ================================================================== */

		template<typename T>
		T get(const YAML::Node& node, const char* key, T def) {
			if (node && node[key]) {
				try { return node[key].template as<T>(); } catch (...) {}
			}
			return def;
		}

		std::string gets(const YAML::Node& node, const char* key, std::string def = {}) {
			return get<std::string>(node, key, std::move(def));
		}

		/* =====================================================================
		 * Sub-config parsers
		 * ================================================================== */

		DatasetConfig parse_dataset(const YAML::Node& n) {
			return {
				.uri                 = gets(n, "uri"),
				.config_name         = gets(n, "config"),
				.split               = gets(n, "split", "train"),
				.batch_size          = get<int64_t>(n, "batch_size", 32),
				.shuffle             = get<bool>(n, "shuffle", true),
				.shuffle_buffer_size = get<int64_t>(n, "shuffle_buffer_size", 10'000),
				.num_workers         = get<int64_t>(n, "num_workers", 4),
				.prefetch            = get<int64_t>(n, "prefetch", 2),
			};
		}

		ValidationConfig parse_validation(const YAML::Node& n) {
			return {
				.uri         = gets(n, "uri"),
				.config_name = gets(n, "config"),
				.split       = gets(n, "split", "validation"),
				.batch_size  = get<int64_t>(n, "batch_size", 32),
			};
		}

		ModelConfig parse_model(const YAML::Node& n) {
			return {
				.path          = gets(n, "path"),
				.function_name = gets(n, "function", "main"),
				.device        = gets(n, "device", "cpu"),
				.device_id     = get<int32_t>(n, "device_id", 0),
			};
		}

		OptimizerConfig parse_optimizer(const YAML::Node& n) {
			return {
				.type         = gets(n, "type", "adamw"),
				.lr           = get<float>(n, "lr",           1e-3f),
				.weight_decay = get<float>(n, "weight_decay", 1e-2f),
				.momentum     = get<float>(n, "momentum",     0.9f),
				.beta1        = get<float>(n, "beta1",        0.9f),
				.beta2        = get<float>(n, "beta2",        0.999f),
				.eps          = get<float>(n, "eps",          1e-8f),
				.nesterov     = get<bool>(n,  "nesterov",     false),
				.amsgrad      = get<bool>(n,  "amsgrad",      false),
			};
		}

		SchedulerConfig parse_scheduler(const YAML::Node& n) {
			return {
				.type         = gets(n, "type",         "cosine_warmup"),
				.warmup_steps = get<int64_t>(n, "warmup_steps", 0),
				.min_lr       = get<float>(n,   "min_lr",       0.0f),
				.step_size    = get<int64_t>(n, "step_size",    1),
				.gamma        = get<float>(n,   "gamma",        0.1f),
			};
		}

		CheckpointConfig parse_checkpoint(const YAML::Node& n) {
			return {
				.dir                 = gets(n, "dir",                  "checkpoints"),
				.save_every_n_epochs = get<int32_t>(n, "save_every_n_epochs", 1),
				.keep_top_k          = get<int32_t>(n, "keep_top_k",          3),
				.monitor             = gets(n, "monitor",              "val_loss"),
				.monitor_mode        = gets(n, "mode",                 "min"),
			};
		}

		TrainingConfig node_to_config(const YAML::Node& root) {
			TrainingConfig cfg;
			cfg.version = gets(root, "version", "1");

			if (const auto n = root["dataset"])    cfg.dataset    = parse_dataset(n);
			if (const auto n = root["validation"]) cfg.validation = parse_validation(n);
			if (const auto n = root["model"])      cfg.model      = parse_model(n);
			if (const auto n = root["optimizer"])  cfg.optimizer  = parse_optimizer(n);
			if (const auto n = root["scheduler"])  cfg.scheduler  = parse_scheduler(n);
			if (const auto n = root["checkpoint"]) cfg.checkpoint = parse_checkpoint(n);

			if (const auto t = root["training"]) {
				cfg.epochs                      = get<int64_t>(t, "epochs",                      10);
				cfg.gradient_accumulation_steps = get<int64_t>(t, "gradient_accumulation_steps", 1);
				cfg.grad_clip_norm              = get<float>(t,   "grad_clip_norm",               0.0f);
				cfg.fp16                        = get<bool>(t,    "fp16",                         false);
				cfg.seed                        = get<int64_t>(t, "seed",                         42);
				cfg.log_level                   = gets(t,         "log_level",                    "info");
			}

			if (const auto ps = root["plugins"]; ps && ps.IsSequence()) {
				for (const auto& p : ps) {
					cfg.plugins.push_back({
						.path   = gets(p, "path"),
						.config = gets(p, "config", "{}"),
					});
				}
			}

			return cfg;
		}

	} // anonymous namespace

	/* =========================================================================
	 * Public API
	 * ====================================================================== */

	std::expected<TrainingConfig, std::string>
	load_config(
		std::span<const std::filesystem::path> files,
		std::span<const std::string>           set_overrides
	) {
		if (files.empty()) {
			return std::unexpected("load_config: no config files provided");
		}

		// YAML::Node default-constructs as Null (not Undefined), so we cannot
		// rely on !merged or operator= to bootstrap the first document.
		// YAML::Clone produces a proper deep copy with correct reference semantics.
		YAML::Node merged;
		bool       first = true;
		for (const auto& file : files) {
			YAML::Node doc;
			try {
				doc = YAML::LoadFile(file.string());
			} catch (const YAML::Exception& e) {
				return std::unexpected(
					std::format("load_config: cannot parse '{}': {}", file.string(), e.what())
				);
			}
			if (first) {
				merged = YAML::Clone(doc);
				first  = false;
			} else {
				deep_merge(merged, doc);
			}
		}

		for (const auto& kv : set_overrides) {
			if (auto r = apply_override(merged, kv); !r) {
				return std::unexpected(r.error());
			}
		}

		interpolate_env(merged);
		return node_to_config(merged);
	}

	std::expected<TrainingConfig, std::string>
	load_config(const std::filesystem::path& file, std::span<const std::string> set_overrides) {
		const std::filesystem::path files[] = {file};
		return load_config(std::span{files}, set_overrides);
	}

} // namespace ttm::conf

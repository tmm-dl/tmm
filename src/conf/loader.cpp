/**
 * @file loader.cpp
 * @brief Config file loading, YAML deep-merge, override application,
 *        environment interpolation, and node → TrainingConfig conversion.
 */

#include <ttm/compat/format.hpp>
#include <ttm/conf/loader.hpp>

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
			if (!overlay.IsMap() || !base.IsMap())
				return;
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
					if (envVal != nullptr)
						result += envVal;
					pos = end + 1;
				}
				if (result != val)
					node = result;
			} else if (node.IsMap()) {
				for (auto kv : node)
					interpolate_env(kv.second);
			} else if (node.IsSequence()) {
				for (auto item : node)
					interpolate_env(item);
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
		static void
		set_nested(YAML::Node node, const std::vector<std::string>& keys, std::size_t idx, const std::string& value) {
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
		std::expected<void, std::string> apply_override(YAML::Node root, std::string_view kv) {
			const auto eq = kv.find('=');
			if (eq == std::string_view::npos) {
				return std::unexpected(std::format("load_config: invalid --set value '{}' (expected key=value)", kv));
			}
			const std::string key_path{kv.substr(0, eq)};
			const std::string value{kv.substr(eq + 1)};

			std::vector<std::string> keys;
			std::istringstream ss{key_path};
			std::string part;
			while (std::getline(ss, part, '.')) {
				if (!part.empty())
					keys.push_back(std::move(part));
			}
			if (keys.empty()) {
				return std::unexpected(std::format("load_config: empty key in --set '{}'", kv));
			}

			set_nested(root, keys, 0, value);
			return {};
		}

		/* =====================================================================
		 * Typed-field extraction helpers
		 * ================================================================== */

		template <typename T>
		T get(const YAML::Node& node, const char* key, T def) {
			if (node && node[key]) {
				try {
					return node[key].template as<T>();
				} catch (...) {
				}
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
					.uri = gets(n, "uri"),
					.config_name = gets(n, "config"),
					.split = gets(n, "split", "train"),
					.batch_size = get<int64_t>(n, "batch_size", 32),
					.shuffle = get<bool>(n, "shuffle", true),
					.shuffle_buffer_size = get<int64_t>(n, "shuffle_buffer_size", 10'000),
					.num_workers = get<int64_t>(n, "num_workers", 4),
					.prefetch = get<int64_t>(n, "prefetch", 2),
			};
		}

		ValidationConfig parse_validation(const YAML::Node& n) {
			return {
					.uri = gets(n, "uri"),
					.config_name = gets(n, "config"),
					.split = gets(n, "split", "validation"),
					.batch_size = get<int64_t>(n, "batch_size", 32),
			};
		}

		ModelConfig parse_model(const YAML::Node& n) {
			return {
					.path = gets(n, "path"),
					.function_name = gets(n, "function", "main"),
					.device = gets(n, "device", "cpu"),
					.device_id = get<int32_t>(n, "device_id", 0),
			};
		}

		OptimizerConfig parse_optimizer(const YAML::Node& n) {
			return {
					.type = gets(n, "type", "adamw"),
					.lr = get<float>(n, "lr", 1e-3f),
					.weight_decay = get<float>(n, "weight_decay", 1e-2f),
					.momentum = get<float>(n, "momentum", 0.9f),
					.beta1 = get<float>(n, "beta1", 0.9f),
					.beta2 = get<float>(n, "beta2", 0.999f),
					.eps = get<float>(n, "eps", 1e-8f),
					.nesterov = get<bool>(n, "nesterov", false),
					.amsgrad = get<bool>(n, "amsgrad", false),
			};
		}

		SchedulerConfig parse_scheduler(const YAML::Node& n) {
			return {
					.type = gets(n, "type", "cosine_warmup"),
					.warmup_steps = get<int64_t>(n, "warmup_steps", 0),
					.min_lr = get<float>(n, "min_lr", 0.0f),
					.step_size = get<int64_t>(n, "step_size", 1),
					.gamma = get<float>(n, "gamma", 0.1f),
					.total_steps = get<int64_t>(n, "total_steps", 0),
			};
		}

		CheckpointConfig parse_checkpoint(const YAML::Node& n) {
			return {
					.dir = gets(n, "dir", "checkpoints"),
					.save_every_n_epochs = get<int32_t>(n, "save_every_n_epochs", 1),
					.keep_top_k = get<int32_t>(n, "keep_top_k", 3),
					.monitor = gets(n, "monitor", "val_loss"),
					.monitor_mode = gets(n, "mode", "min"),
			};
		}

		/* =====================================================================
		 * New-schema helpers
		 * ================================================================== */

		/**
		 * @brief Serialize a YAML node to a JSON string.
		 *
		 * @details
		 * Used to build the `config` JSON for callback, preprocessor, and other
		 * plugin entries from their YAML subtrees without custom per-type parsing.
		 * Scalars are auto-detected as null / bool / number / string.
		 */
		std::string yaml_to_json(const YAML::Node& n) {
			switch (n.Type()) {
			case YAML::NodeType::Null:
				return "null";
			case YAML::NodeType::Scalar: {
				const std::string s = n.as<std::string>();
				if (s == "null")
					return "null";
				if (s == "true")
					return "true";
				if (s == "false")
					return "false";
				// Detect numeric scalars: try integer first, then float
				try {
					(void)std::stoll(s);
					return s;
				} catch (...) {
				}
				try {
					(void)std::stod(s);
					return s;
				} catch (...) {
				}
				// String: escape backslash, double-quote, newline
				std::string out;
				out.reserve(s.size() + 2);
				out += '"';
				for (char c : s) {
					if (c == '"') {
						out += "\\\"";
					} else if (c == '\\') {
						out += "\\\\";
					} else if (c == '\n') {
						out += "\\n";
					} else if (c == '\r') {
						out += "\\r";
					} else {
						out += c;
					}
				}
				out += '"';
				return out;
			}
			case YAML::NodeType::Sequence: {
				std::string out = "[";
				bool first = true;
				for (const auto& item : n) {
					if (!first)
						out += ',';
					out += yaml_to_json(item);
					first = false;
				}
				return out + ']';
			}
			case YAML::NodeType::Map: {
				std::string out = "{";
				bool first = true;
				for (const auto& kv : n) {
					if (!first)
						out += ',';
					out += '"' + kv.first.as<std::string>() + "\":" + yaml_to_json(kv.second);
					first = false;
				}
				return out + '}';
			}
			default:
				return "null";
			}
		}

		TrainingConfig node_to_config(const YAML::Node& root, const std::filesystem::path& config_dir) {
			TrainingConfig cfg;
			cfg.version = gets(root, "version", "1");

			/* ------------------------------------------------------------------
			 * New schema: data.train.* / data.validation.*
			 * Old schema: dataset.* / validation.*  (still supported as fallback)
			 * ---------------------------------------------------------------- */
			if (const auto data = root["data"]) {
				if (const auto n = data["train"]) {
					cfg.dataset.uri = gets(n, "url", gets(n, "uri"));
					cfg.dataset.split = gets(n, "split", "train");
					cfg.dataset.batch_size = get<int64_t>(n, "batch_size", 32);
					cfg.dataset.shuffle = get<bool>(n, "shuffle", true);
					cfg.dataset.shuffle_buffer_size = get<int64_t>(n, "shuffle_buffer_size", 10'000);
					cfg.dataset.num_workers = get<int64_t>(n, "num_workers", 4);
					cfg.dataset.prefetch = get<int64_t>(n, "prefetch", 2);

					// Per-dataset preprocessors: data.train.preprocessor[]
					if (const auto pps = n["preprocessor"]; pps && pps.IsSequence()) {
						for (const auto& p : pps) {
							cfg.preprocessors.push_back({
									.type = gets(p, "type", gets(p, "name")),
									.config = gets(p, "config", "{}"),
							});
						}
					}
				}
				if (const auto n = data["validation"]) {
					ValidationConfig vc;
					vc.uri = gets(n, "url", gets(n, "uri", cfg.dataset.uri.empty() ? "" : cfg.dataset.uri));
					vc.split = gets(n, "split", "validation");
					vc.batch_size = get<int64_t>(n, "batch_size", 32);
					cfg.validation = vc;
				}
			} else {
				// Old schema fallback
				if (const auto n = root["dataset"])
					cfg.dataset = parse_dataset(n);
				if (const auto n = root["validation"])
					cfg.validation = parse_validation(n);
			}

			/* ------------------------------------------------------------------
			 * Model: new schema uses model.file (relative path) + model.device.
			 * Old schema uses model.path.
			 * ---------------------------------------------------------------- */
			if (const auto n = root["model"]) {
				const std::string file = gets(n, "file");
				const std::string path = gets(n, "path");
				if (!file.empty()) {
					// Resolve relative to the config file's directory
					cfg.model.path = (config_dir / file).string();
				} else {
					cfg.model.path = path;
				}
				cfg.model.function_name = gets(n, "function", "main");
				cfg.model.device = gets(n, "device", "cpu");
				cfg.model.device_id = get<int32_t>(n, "device_id", 0);
			}

			/* ------------------------------------------------------------------
			 * Trainer block (new schema): trainer.optimizer, trainer.lr_scheduler,
			 * trainer.early_stopping, trainer.checkpoint, trainer.epochs, …
			 * Old schema: optimizer.*, scheduler.*, checkpoint.*, training.*
			 * ---------------------------------------------------------------- */
			if (const auto t = root["trainer"]) {
				cfg.epochs = get<int64_t>(t, "epochs", 10);
				cfg.gradient_accumulation_steps = get<int64_t>(t, "gradient_accumulation_steps", 1);

				if (const auto o = t["optimizer"]) {
					cfg.optimizer.type = gets(o, "type", "adamw");
					cfg.optimizer.lr = get<float>(o, "lr", 1e-3f);
					cfg.optimizer.weight_decay = get<float>(o, "weight_decay", 1e-2f);
					cfg.optimizer.beta1 = get<float>(o, "beta1", 0.9f);
					cfg.optimizer.beta2 = get<float>(o, "beta2", 0.999f);
					cfg.optimizer.eps = get<float>(o, "eps", 1e-8f);
					cfg.optimizer.amsgrad = get<bool>(o, "amsgrad", false);
				}
				if (const auto s = t["lr_scheduler"]) {
					cfg.scheduler.type = gets(s, "type", "cosine_warmup");
					cfg.scheduler.warmup_steps = get<int64_t>(s, "warmup_steps", 0);
					cfg.scheduler.min_lr = get<float>(s, "min_lr", 0.0f);
					cfg.scheduler.step_size = get<int64_t>(s, "step_size", 1);
					cfg.scheduler.gamma = get<float>(s, "gamma", 0.1f);
					cfg.scheduler.total_steps = get<int64_t>(s, "total_steps", 0);
				}
				if (const auto es = t["early_stopping"]) {
					cfg.callbacks.push_back({
							.type = "early_stopping",
							.config = yaml_to_json(es),
					});
				}
				if (const auto ck = t["checkpoint"]) {
					cfg.checkpoint.dir = gets(ck, "directory", gets(ck, "dir", "checkpoints"));
					cfg.checkpoint.save_every_n_epochs =
							get<int32_t>(ck, "every_n_epochs", get<int32_t>(ck, "save_every_n_epochs", 1));
					cfg.checkpoint.keep_top_k = get<int32_t>(ck, "keep_top_k", 3);
					cfg.checkpoint.monitor = gets(ck, "monitor", "val_loss");
					cfg.checkpoint.monitor_mode = gets(ck, "mode", "min");
					cfg.callbacks.push_back({
							.type = "checkpoint",
							.config = yaml_to_json(ck),
					});
				}
				// New schema: trainer.callbacks[] (takes priority over the above shorthands)
				if (const auto cbs = t["callbacks"]; cbs && cbs.IsSequence()) {
					for (const auto& c : cbs) {
						cfg.callbacks.push_back({
								.type = gets(c, "type", gets(c, "name")),
								.config = yaml_to_json(c),
						});
					}
				}
			} else {
				// Old schema fallback
				if (const auto n = root["optimizer"])
					cfg.optimizer = parse_optimizer(n);
				if (const auto n = root["scheduler"])
					cfg.scheduler = parse_scheduler(n);
				if (const auto n = root["checkpoint"])
					cfg.checkpoint = parse_checkpoint(n);
				if (const auto tr = root["training"]) {
					cfg.epochs = get<int64_t>(tr, "epochs", 10);
					cfg.gradient_accumulation_steps = get<int64_t>(tr, "gradient_accumulation_steps", 1);
					cfg.grad_clip_norm = get<float>(tr, "grad_clip_norm", 0.0f);
					cfg.fp16 = get<bool>(tr, "fp16", false);
					cfg.seed = get<int64_t>(tr, "seed", 42);
					cfg.log_level = gets(tr, "log_level", "info");
				}
			}

			/* ------------------------------------------------------------------
			 * Plugins: support both name: and path: forms.
			 * ---------------------------------------------------------------- */
			if (const auto ps = root["plugins"]; ps && ps.IsSequence()) {
				for (const auto& p : ps) {
					cfg.plugins.push_back({
							.name = gets(p, "name"),
							.path = gets(p, "path"),
							.config = gets(p, "config", "{}"),
							.optional = get<bool>(p, "optional", false),
					});
				}
			}

			/* ------------------------------------------------------------------
			 * Top-level preprocessors[] (old schema; new schema puts them under
			 * data.train.preprocessor[] which is already handled above).
			 * ---------------------------------------------------------------- */
			if (const auto pps = root["preprocessors"]; pps && pps.IsSequence()) {
				for (const auto& p : pps) {
					cfg.preprocessors.push_back({
							.type = gets(p, "type"),
							.config = gets(p, "config", "{}"),
					});
				}
			}

			/* ------------------------------------------------------------------
			 * Top-level callbacks[] (old schema / standalone use).
			 * ---------------------------------------------------------------- */
			if (const auto cs = root["callbacks"]; cs && cs.IsSequence()) {
				for (const auto& c : cs) {
					cfg.callbacks.push_back({
							.type = gets(c, "type", gets(c, "name")),
							.config = yaml_to_json(c),
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
	load_config(std::span<const std::filesystem::path> files, std::span<const std::string> set_overrides) {
		if (files.empty()) {
			return std::unexpected("load_config: no config files provided");
		}

		// The config directory is the directory of the first (primary) config file.
		// model.file paths are resolved relative to it.
		const std::filesystem::path config_dir = std::filesystem::absolute(files[0]).parent_path();

		// YAML::Node default-constructs as Null (not Undefined), so we cannot
		// rely on !merged or operator= to bootstrap the first document.
		// YAML::Clone produces a proper deep copy with correct reference semantics.
		YAML::Node merged;
		bool first = true;
		for (const auto& file : files) {
			YAML::Node doc;
			try {
				doc = YAML::LoadFile(file.string());
			} catch (const YAML::Exception& e) {
				return std::unexpected(std::format("load_config: cannot parse '{}': {}", file.string(), e.what()));
			}
			if (first) {
				merged = YAML::Clone(doc);
				first = false;
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
		return node_to_config(merged, config_dir);
	}

	std::expected<TrainingConfig, std::string>
	load_config(const std::filesystem::path& file, std::span<const std::string> set_overrides) {
		const std::filesystem::path files[] = {file};
		return load_config(std::span{files}, set_overrides);
	}

} // namespace ttm::conf

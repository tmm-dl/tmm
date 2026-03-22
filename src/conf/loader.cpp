/**
 * @file loader.cpp
 * @brief Config file loading, YAML deep-merge, override application,
 *        environment interpolation, and node → TrainingConfig conversion.
 */

#include <tmm/compat/format.hpp>
#include <tmm/conf/loader.hpp>

#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tmm::conf {

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
		void deepMerge(YAML::Node base, const YAML::Node& overlay) {
			if (!overlay.IsMap() || !base.IsMap())
				return;
			for (const auto& kv : overlay) {
				const std::string key = kv.first.as<std::string>();
				if (base[key] && base[key].IsMap() && kv.second.IsMap()) {
					deepMerge(base[key], kv.second);
				} else {
					base[key] = kv.second;
				}
			}
		}

		/**
		 * @brief Walk all scalar string nodes and expand `${VAR_NAME}` env references.
		 */
		void interpolateEnv(YAML::Node node) {
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
					interpolateEnv(kv.second);
			} else if (node.IsSequence()) {
				for (auto item : node)
					interpolateEnv(item);
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
		setNested(YAML::Node node, const std::vector<std::string>& keys, std::size_t idx, const std::string& value) {
			if (idx + 1 == keys.size()) {
				node[keys[idx]] = value;
				return;
			}
			setNested(node[keys[idx]], keys, idx + 1, value);
		}

		/**
		 * @brief Apply a single `"key.path=value"` override to the YAML root node.
		 *
		 * Intermediate mapping nodes are created automatically.  The leaf is always
		 * written as a YAML scalar string; yaml-cpp converts to the target type when
		 * read via `.as<T>()`.
		 */
		std::expected<void, std::string> applyOverride(YAML::Node root, std::string_view kv) {
			const auto eq = kv.find('=');
			if (eq == std::string_view::npos) {
				return std::unexpected(std::format("load_config: invalid --set value '{}' (expected key=value)", kv));
			}
			const std::string keyPath{kv.substr(0, eq)};
			const std::string value{kv.substr(eq + 1)};

			std::vector<std::string> keys;
			std::istringstream ss{keyPath};
			std::string part;
			while (std::getline(ss, part, '.')) {
				if (!part.empty())
					keys.push_back(std::move(part));
			}
			if (keys.empty()) {
				return std::unexpected(std::format("load_config: empty key in --set '{}'", kv));
			}

			setNested(root, keys, 0, value);
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

		DatasetConfig parseDataset(const YAML::Node& n) {
			return {
					.uri = gets(n, "uri"),
					.configName = gets(n, "config"),
					.split = gets(n, "split", "train"),
					.batchSize = get<int64_t>(n, "batch_size", 32),
					.shuffle = get<bool>(n, "shuffle", true),
					.shuffleBufferSize = get<int64_t>(n, "shuffle_buffer_size", 10'000),
					.numWorkers = get<int64_t>(n, "num_workers", 4),
					.prefetch = get<int64_t>(n, "prefetch", 2),
			};
		}

		ValidationConfig parseValidation(const YAML::Node& n) {
			return {
					.uri = gets(n, "uri"),
					.configName = gets(n, "config"),
					.split = gets(n, "split", "validation"),
					.batchSize = get<int64_t>(n, "batch_size", 32),
			};
		}

		ModelConfig parseModel(const YAML::Node& n) {
			return {
					.path = gets(n, "path"),
					.functionName = gets(n, "function", "main"),
					.device = gets(n, "device", "cpu"),
					.deviceId = get<int32_t>(n, "device_id", 0),
			};
		}

		OptimizerConfig parseOptimizer(const YAML::Node& n) {
			return {
					.type = gets(n, "type", "adamw"),
					.lr = get<float>(n, "lr", 1e-3f),
					.weightDecay = get<float>(n, "weight_decay", 1e-2f),
					.momentum = get<float>(n, "momentum", 0.9f),
					.beta1 = get<float>(n, "beta1", 0.9f),
					.beta2 = get<float>(n, "beta2", 0.999f),
					.eps = get<float>(n, "eps", 1e-8f),
					.nesterov = get<bool>(n, "nesterov", false),
					.amsgrad = get<bool>(n, "amsgrad", false),
			};
		}

		SchedulerConfig parseScheduler(const YAML::Node& n) {
			return {
					.type = gets(n, "type", "cosine_warmup"),
					.warmupSteps = get<int64_t>(n, "warmup_steps", 0),
					.minLr = get<float>(n, "min_lr", 0.0f),
					.stepSize = get<int64_t>(n, "step_size", 1),
					.gamma = get<float>(n, "gamma", 0.1f),
					.totalSteps = get<int64_t>(n, "total_steps", 0),
			};
		}

		CheckpointConfig parseCheckpoint(const YAML::Node& n) {
			return {
					.dir = gets(n, "dir", "checkpoints"),
					.saveEveryNEpochs = get<int32_t>(n, "save_every_n_epochs", 1),
					.keepTopK = get<int32_t>(n, "keep_top_k", 3),
					.monitor = gets(n, "monitor", "val_loss"),
					.monitorMode = gets(n, "mode", "min"),
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
		std::string yamlToJson(const YAML::Node& n) {
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
					out += yamlToJson(item);
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
					out += '"' + kv.first.as<std::string>() + "\":" + yamlToJson(kv.second);
					first = false;
				}
				return out + '}';
			}
			default:
				return "null";
			}
		}

		TrainingConfig nodeToConfig(const YAML::Node& root, const std::filesystem::path& configDir) {
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
					cfg.dataset.batchSize = get<int64_t>(n, "batch_size", 32);
					cfg.dataset.shuffle = get<bool>(n, "shuffle", true);
					cfg.dataset.shuffleBufferSize = get<int64_t>(n, "shuffle_buffer_size", 10'000);
					cfg.dataset.numWorkers = get<int64_t>(n, "num_workers", 4);
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
					vc.batchSize = get<int64_t>(n, "batch_size", 32);
					cfg.validation = vc;
				}
			} else {
				// Old schema fallback
				if (const auto n = root["dataset"])
					cfg.dataset = parseDataset(n);
				if (const auto n = root["validation"])
					cfg.validation = parseValidation(n);
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
					cfg.model.path = (configDir / file).string();
				} else {
					cfg.model.path = path;
				}
				cfg.model.functionName = gets(n, "function", "main");
				cfg.model.device = gets(n, "device", "cpu");
				cfg.model.deviceId = get<int32_t>(n, "device_id", 0);
			}

			/* ------------------------------------------------------------------
			 * Trainer block (new schema): trainer.optimizer, trainer.lr_scheduler,
			 * trainer.early_stopping, trainer.checkpoint, trainer.epochs, …
			 * Old schema: optimizer.*, scheduler.*, checkpoint.*, training.*
			 * ---------------------------------------------------------------- */
			if (const auto t = root["trainer"]) {
				cfg.epochs = get<int64_t>(t, "epochs", 10);
				cfg.gradientAccumulationSteps = get<int64_t>(t, "gradient_accumulation_steps", 1);

				if (const auto o = t["optimizer"]) {
					cfg.optimizer.type = gets(o, "type", "adamw");
					cfg.optimizer.lr = get<float>(o, "lr", 1e-3f);
					cfg.optimizer.weightDecay = get<float>(o, "weight_decay", 1e-2f);
					cfg.optimizer.beta1 = get<float>(o, "beta1", 0.9f);
					cfg.optimizer.beta2 = get<float>(o, "beta2", 0.999f);
					cfg.optimizer.eps = get<float>(o, "eps", 1e-8f);
					cfg.optimizer.amsgrad = get<bool>(o, "amsgrad", false);
				}
				if (const auto s = t["lr_scheduler"]) {
					cfg.scheduler.type = gets(s, "type", "cosine_warmup");
					cfg.scheduler.warmupSteps = get<int64_t>(s, "warmup_steps", 0);
					cfg.scheduler.minLr = get<float>(s, "min_lr", 0.0f);
					cfg.scheduler.stepSize = get<int64_t>(s, "step_size", 1);
					cfg.scheduler.gamma = get<float>(s, "gamma", 0.1f);
					cfg.scheduler.totalSteps = get<int64_t>(s, "total_steps", 0);
				}
				if (const auto es = t["early_stopping"]) {
					cfg.callbacks.push_back({
							.type = "early_stopping",
							.config = yamlToJson(es),
					});
				}
				if (const auto ck = t["checkpoint"]) {
					cfg.checkpoint.dir = gets(ck, "directory", gets(ck, "dir", "checkpoints"));
					cfg.checkpoint.saveEveryNEpochs =
							get<int32_t>(ck, "every_n_epochs", get<int32_t>(ck, "save_every_n_epochs", 1));
					cfg.checkpoint.keepTopK = get<int32_t>(ck, "keep_top_k", 3);
					cfg.checkpoint.monitor = gets(ck, "monitor", "val_loss");
					cfg.checkpoint.monitorMode = gets(ck, "mode", "min");
					cfg.callbacks.push_back({
							.type = "checkpoint",
							.config = yamlToJson(ck),
					});
				}
				// New schema: trainer.callbacks[] (takes priority over the above shorthands)
				if (const auto cbs = t["callbacks"]; cbs && cbs.IsSequence()) {
					for (const auto& c : cbs) {
						cfg.callbacks.push_back({
								.type = gets(c, "type", gets(c, "name")),
								.config = yamlToJson(c),
						});
					}
				}
			} else {
				// Old schema fallback
				if (const auto n = root["optimizer"])
					cfg.optimizer = parseOptimizer(n);
				if (const auto n = root["scheduler"])
					cfg.scheduler = parseScheduler(n);
				if (const auto n = root["checkpoint"])
					cfg.checkpoint = parseCheckpoint(n);
				if (const auto tr = root["training"]) {
					cfg.epochs = get<int64_t>(tr, "epochs", 10);
					cfg.gradientAccumulationSteps = get<int64_t>(tr, "gradient_accumulation_steps", 1);
					cfg.gradClipNorm = get<float>(tr, "grad_clip_norm", 0.0f);
					cfg.fp16 = get<bool>(tr, "fp16", false);
					cfg.seed = get<int64_t>(tr, "seed", 42);
					cfg.logLevel = gets(tr, "log_level", "info");
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
							.config = yamlToJson(c),
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
		const std::filesystem::path configDir = std::filesystem::absolute(files[0]).parent_path();

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
				deepMerge(merged, doc);
			}
		}

		for (const auto& kv : set_overrides) {
			if (auto r = applyOverride(merged, kv); !r) {
				return std::unexpected(r.error());
			}
		}

		interpolateEnv(merged);
		return nodeToConfig(merged, configDir);
	}

	std::expected<TrainingConfig, std::string>
	load_config(const std::filesystem::path& file, std::span<const std::string> set_overrides) {
		const std::filesystem::path files[] = {file};
		return load_config(std::span{files}, set_overrides);
	}

} // namespace tmm::conf

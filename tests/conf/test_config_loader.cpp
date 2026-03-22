/**
 * @file test_config_loader.cpp
 * @brief Unit tests for tmm::conf::load_config — deep-merge, override
 *        application, env interpolation, and both YAML schemas.
 *
 * @details
 * All tests write temporary YAML files so that load_config() can be exercised
 * end-to-end through its public API.  The internal helpers (deepMerge,
 * applyOverride, interpolateEnv, nodeToConfig) live in an anonymous namespace
 * and are therefore only reachable indirectly.
 */

#include <tmm/conf/loader.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// =============================================================================
// Helpers
// =============================================================================

namespace {

	/// Write `content` to a temporary file and return its path.
	std::filesystem::path writeTmp(std::string_view content, std::string_view suffix = ".yml") {
		static std::atomic<int> counter{0};
		auto path = std::filesystem::temp_directory_path() /
					("tmm_conf_test_" + std::to_string(counter++) + std::string(suffix));
		std::ofstream ofs(path);
		ofs << content;
		return path;
	}

	/// RAII guard that removes a file on destruction.
	struct AutoRemove {
		std::vector<std::filesystem::path> paths;
		~AutoRemove() {
			for (const auto& p : paths)
				std::filesystem::remove(p);
		}
	};

} // anonymous namespace

// =============================================================================
// Error handling
// =============================================================================

TEST_CASE("load_config returns error for empty file list", "[conf][error]") {
	auto result = tmm::conf::load_config(std::span<const std::filesystem::path>{});
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error().find("no config files") != std::string::npos);
}

TEST_CASE("load_config returns error for missing file", "[conf][error]") {
	const std::filesystem::path missing = "/tmp/this_file_does_not_exist_ttm.yml";
	auto result = tmm::conf::load_config(missing);
	REQUIRE_FALSE(result.has_value());
}

TEST_CASE("load_config returns error for invalid YAML", "[conf][error]") {
	AutoRemove guard;
	auto path = writeTmp("{ invalid: yaml: content: [\n");
	guard.paths.push_back(path);
	auto result = tmm::conf::load_config(path);
	REQUIRE_FALSE(result.has_value());
}

TEST_CASE("load_config returns error for invalid --set syntax (no equals)", "[conf][error]") {
	AutoRemove guard;
	auto path = writeTmp("version: \"1\"\n");
	guard.paths.push_back(path);

	std::vector<std::string> overrides = {"trainer.epochs"};
	auto result = tmm::conf::load_config(path, overrides);
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error().find("invalid --set") != std::string::npos);
}

// =============================================================================
// Default values
// =============================================================================

TEST_CASE("load_config applies default values for an empty document", "[conf][defaults]") {
	AutoRemove guard;
	auto path = writeTmp("version: \"1\"\n");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	CHECK(cfg.version == "1");
	CHECK(cfg.epochs == 10);
	CHECK(cfg.gradientAccumulationSteps == 1);
	CHECK(cfg.optimizer.type == "adamw");
	CHECK(cfg.optimizer.lr == Catch::Approx(1e-3f));
	CHECK(cfg.optimizer.weightDecay == Catch::Approx(1e-2f));
	CHECK(cfg.scheduler.type == "cosine_warmup");
	CHECK(cfg.model.functionName == "main");
	CHECK(cfg.model.device == "cpu");
	CHECK(cfg.model.deviceId == 0);
	CHECK(cfg.dataset.split == "train");
	CHECK(cfg.dataset.batchSize == 32);
	CHECK(cfg.dataset.shuffle == true);
	CHECK_FALSE(cfg.validation.has_value());
	CHECK(cfg.plugins.empty());
	CHECK(cfg.preprocessors.empty());
	CHECK(cfg.callbacks.empty());
}

// =============================================================================
// New schema (data.train / trainer)
// =============================================================================

TEST_CASE("load_config parses new-schema data.train block", "[conf][new-schema]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
data:
  train:
    url: hf:owner/dataset
    split: train
    batch_size: 16
    shuffle: false
    num_workers: 2
)");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	CHECK(cfg.dataset.uri == "hf:owner/dataset");
	CHECK(cfg.dataset.split == "train");
	CHECK(cfg.dataset.batchSize == 16);
	CHECK(cfg.dataset.shuffle == false);
	CHECK(cfg.dataset.numWorkers == 2);
}

TEST_CASE("load_config parses new-schema data.validation block", "[conf][new-schema]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
data:
  train:
    url: hf:owner/dataset
  validation:
    url: hf:owner/dataset
    split: validation
    batch_size: 64
)");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	REQUIRE(cfg.validation.has_value());
	CHECK(cfg.validation->uri == "hf:owner/dataset");
	CHECK(cfg.validation->split == "validation");
	CHECK(cfg.validation->batchSize == 64);
}

TEST_CASE("load_config parses new-schema trainer block", "[conf][new-schema]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
trainer:
  epochs: 5
  gradient_accumulation_steps: 4
  optimizer:
    type: adamw
    lr: 3.0e-4
    weight_decay: 1.0e-2
    beta1: 0.9
    beta2: 0.999
    eps: 1.0e-8
  lr_scheduler:
    type: cosine_warmup
    warmup_steps: 100
    min_lr: 1.0e-6
)");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	CHECK(cfg.epochs == 5);
	CHECK(cfg.gradientAccumulationSteps == 4);
	CHECK(cfg.optimizer.type == "adamw");
	CHECK(cfg.optimizer.lr == Catch::Approx(3e-4f));
	CHECK(cfg.optimizer.weightDecay == Catch::Approx(1e-2f));
	CHECK(cfg.optimizer.beta1 == Catch::Approx(0.9f));
	CHECK(cfg.optimizer.beta2 == Catch::Approx(0.999f));
	CHECK(cfg.optimizer.eps == Catch::Approx(1e-8f));
	CHECK(cfg.scheduler.type == "cosine_warmup");
	CHECK(cfg.scheduler.warmupSteps == 100);
	CHECK(cfg.scheduler.minLr == Catch::Approx(1e-6f));
}

TEST_CASE("load_config parses new-schema model.file (relative path)", "[conf][new-schema]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
model:
  file: model.so
  device: cuda
  device_id: 1
)");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	// model.file resolved relative to config directory
	const auto expectedPath = (path.parent_path() / "model.so").string();
	CHECK(cfg.model.path == expectedPath);
	CHECK(cfg.model.device == "cuda");
	CHECK(cfg.model.deviceId == 1);
}

TEST_CASE("load_config parses new-schema preprocessors under data.train", "[conf][new-schema]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
data:
  train:
    url: hf:owner/dataset
    preprocessor:
      - type: bpe-tokenize
        config: '{"vocab":"vocab.json","max_length":512}'
      - type: truncate
        config: '{}'
)");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	REQUIRE(cfg.preprocessors.size() == 2);
	CHECK(cfg.preprocessors[0].type == "bpe-tokenize");
	CHECK(cfg.preprocessors[0].config == R"({"vocab":"vocab.json","max_length":512})");
	CHECK(cfg.preprocessors[1].type == "truncate");
}

// =============================================================================
// Legacy schema (dataset / optimizer / training)
// =============================================================================

TEST_CASE("load_config parses legacy dataset block", "[conf][legacy-schema]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
dataset:
  uri: hf:thagen/SCITE
  config: causality detection
  split: train
  batch_size: 32
)");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	CHECK(cfg.dataset.uri == "hf:thagen/SCITE");
	CHECK(cfg.dataset.configName == "causality detection");
	CHECK(cfg.dataset.split == "train");
	CHECK(cfg.dataset.batchSize == 32);
}

TEST_CASE("load_config parses legacy optimizer + training blocks", "[conf][legacy-schema]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
optimizer:
  type: sgd
  lr: 1.0e-2
  momentum: 0.95

training:
  epochs: 20
  gradient_accumulation_steps: 8
  grad_clip_norm: 1.0
  seed: 123
)");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	CHECK(cfg.optimizer.type == "sgd");
	CHECK(cfg.optimizer.lr == Catch::Approx(1e-2f));
	CHECK(cfg.optimizer.momentum == Catch::Approx(0.95f));
	CHECK(cfg.epochs == 20);
	CHECK(cfg.gradientAccumulationSteps == 8);
	CHECK(cfg.gradClipNorm == Catch::Approx(1.0f));
	CHECK(cfg.seed == 123);
}

// =============================================================================
// deepMerge (tested via multi-file load)
// =============================================================================

TEST_CASE("load_config deep-merges two files: scalars are overridden by right file", "[conf][merge]") {
	AutoRemove guard;
	auto base = writeTmp(R"(
trainer:
  epochs: 10
  optimizer:
    type: adamw
    lr: 1.0e-3
)");
	auto overlay = writeTmp(R"(
trainer:
  epochs: 5
  optimizer:
    lr: 3.0e-4
)");
	guard.paths = {base, overlay};

	const std::filesystem::path files[] = {base, overlay};
	auto result = tmm::conf::load_config(std::span{files});
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	// epochs overridden by overlay
	CHECK(cfg.epochs == 5);
	// optimizer.type preserved from base (map merge, not replace)
	CHECK(cfg.optimizer.type == "adamw");
	// optimizer.lr overridden by overlay
	CHECK(cfg.optimizer.lr == Catch::Approx(3e-4f));
}

TEST_CASE("load_config deep-merges: nested map keys are preserved from base when overlay omits them", "[conf][merge]") {
	AutoRemove guard;
	auto base = writeTmp(R"(
trainer:
  optimizer:
    type: adamw
    lr: 1.0e-3
    weight_decay: 1.0e-2
    beta1: 0.9
    beta2: 0.999
)");
	auto overlay = writeTmp(R"(
trainer:
  optimizer:
    lr: 5.0e-5
)");
	guard.paths = {base, overlay};

	const std::filesystem::path files[] = {base, overlay};
	auto result = tmm::conf::load_config(std::span{files});
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	CHECK(cfg.optimizer.type == "adamw");					  // preserved from base
	CHECK(cfg.optimizer.lr == Catch::Approx(5e-5f));		  // overridden
	CHECK(cfg.optimizer.weightDecay == Catch::Approx(1e-2f)); // preserved
	CHECK(cfg.optimizer.beta1 == Catch::Approx(0.9f));		  // preserved
}

TEST_CASE("load_config deep-merges three files left-to-right", "[conf][merge]") {
	AutoRemove guard;
	auto f1 = writeTmp("trainer:\n  epochs: 1\n");
	auto f2 = writeTmp("trainer:\n  epochs: 2\n");
	auto f3 = writeTmp("trainer:\n  epochs: 3\n");
	guard.paths = {f1, f2, f3};

	const std::filesystem::path files[] = {f1, f2, f3};
	auto result = tmm::conf::load_config(std::span{files});
	REQUIRE(result.has_value());
	CHECK(result->epochs == 3);
}

// =============================================================================
// applyOverride (tested via --set overrides)
// =============================================================================

TEST_CASE("load_config applies --set scalar override", "[conf][override]") {
	AutoRemove guard;
	auto path = writeTmp("trainer:\n  epochs: 10\n");
	guard.paths.push_back(path);

	std::vector<std::string> overrides = {"trainer.epochs=3"};
	auto result = tmm::conf::load_config(path, overrides);
	REQUIRE(result.has_value());
	CHECK(result->epochs == 3);
}

TEST_CASE("load_config applies --set nested override creating missing intermediate keys", "[conf][override]") {
	AutoRemove guard;
	auto path = writeTmp("version: \"1\"\n");
	guard.paths.push_back(path);

	std::vector<std::string> overrides = {"trainer.optimizer.lr=5e-5"};
	auto result = tmm::conf::load_config(path, overrides);
	REQUIRE(result.has_value());
	CHECK(result->optimizer.lr == Catch::Approx(5e-5f));
}

TEST_CASE("load_config applies multiple --set overrides in order", "[conf][override]") {
	AutoRemove guard;
	auto path = writeTmp("trainer:\n  epochs: 10\n");
	guard.paths.push_back(path);

	std::vector<std::string> overrides = {"trainer.epochs=5", "trainer.epochs=2"};
	auto result = tmm::conf::load_config(path, overrides);
	REQUIRE(result.has_value());
	CHECK(result->epochs == 2);
}

// =============================================================================
// interpolateEnv (tested via ${VAR} in YAML)
// =============================================================================

TEST_CASE("load_config expands ${ENV_VAR} in scalar strings", "[conf][env]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
model:
  path: /models/${MODEL_FILE}
)");
	guard.paths.push_back(path);

#ifdef _WIN32
	_putenv_s("MODEL_FILE", "gpt2.so");
#else
	setenv("MODEL_FILE", "gpt2.so", 1);
#endif

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	CHECK(result->model.path == "/models/gpt2.so");

#ifdef _WIN32
	_putenv_s("MODEL_FILE", "");
#else
	unsetenv("MODEL_FILE");
#endif
}

TEST_CASE("load_config leaves ${UNSET_VAR} as empty string", "[conf][env]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
model:
  path: /base/${TTMTEST_UNSET_VAR_12345}/model.so
)");
	guard.paths.push_back(path);

	// Ensure the variable is not set
#ifdef _WIN32
	_putenv_s("TTMTEST_UNSET_VAR_12345", "");
#else
	unsetenv("TTMTEST_UNSET_VAR_12345");
#endif

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	CHECK(result->model.path == "/base//model.so");
}

// =============================================================================
// Plugins block
// =============================================================================

TEST_CASE("load_config parses plugins by name", "[conf][plugins]") {
	AutoRemove guard;
	auto path = writeTmp(R"(
plugins:
  - name: core
  - name: python
    config: '{"key":"value"}'
  - path: /opt/myplugin.wasm
    optional: true
)");
	guard.paths.push_back(path);

	auto result = tmm::conf::load_config(path);
	REQUIRE(result.has_value());
	const auto& cfg = *result;

	REQUIRE(cfg.plugins.size() == 3);
	CHECK(cfg.plugins[0].name == "core");
	CHECK(cfg.plugins[0].config == "{}");
	CHECK(cfg.plugins[0].optional == false);
	CHECK(cfg.plugins[1].name == "python");
	CHECK(cfg.plugins[1].config == R"({"key":"value"})");
	CHECK(cfg.plugins[2].path == "/opt/myplugin.wasm");
	CHECK(cfg.plugins[2].optional == true);
}

/**
 * @file main.cpp
 * @brief ttm — Train My Model — CLI entry point.
 *
 * ### Commands
 * - `ttm fit <config.yml> [config2.yml …] [--set key=value …]`
 *   Load config, set up plugins, and run the training loop.
 *
 * - `ttm validate <config.yml> [--set key=value …]`
 *   Parse and print the resolved config without training.
 *
 * - `ttm predict`  *(stub — not yet implemented)*
 *
 * ### Config format (YAML)
 * @code{.yaml}
 * version: "1"
 *
 * dataset:
 *   uri:    hf:thagen/SCITE
 *   config: causality detection
 *   split:  train
 *   batch_size: 32
 *
 * model:
 *   path:   ./model.so
 *   device: cpu
 *
 * optimizer:
 *   type: adamw
 *   lr:   1.0e-4
 *
 * scheduler:
 *   type:         cosine_warmup
 *   warmup_steps: 100
 *
 * training:
 *   epochs: 10
 *   gradient_accumulation_steps: 4
 *   grad_clip_norm: 1.0
 *
 * plugins:
 *   - path:   plugins/csv-logger.wasm
 *     config: '{"output":"metrics.csv"}'
 * @endcode
 */

#include <ttm/conf/config.hpp>
#include <ttm/conf/loader.hpp>
#include <ttm/datasets/dataset_loader.hpp>
#include <ttm/model/device.hpp>
#include <ttm/model/pipeline.hpp>
#include <ttm/plugins/plugin_manager.hpp>
#include <ttm/trainer/trainer.hpp>

#include <CLI/CLI.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

/* =========================================================================
 * Helpers
 * ====================================================================== */

/// Extract the URI scheme (everything up to and including the first `:`).
static std::string scheme_of(const std::string& uri) {
	const auto colon = uri.find(':');
	return (colon != std::string::npos) ? uri.substr(0, colon + 1) : "file:";
}

/**
 * @brief ILRScheduler backed by a plugin-registered ttm_scheduler_vtable.
 *
 * Wraps the C vtable so that PluginManager::find_scheduler_vtable() results
 * can be attached to a Trainer without exposing the C ABI above this file.
 */
class VtableScheduler final : public ttm::trainer::ILRScheduler {
public:
	VtableScheduler(const ttm_scheduler_vtable& vt, ttm_handle h) : vt_(vt), h_(h) {}

	~VtableScheduler() override {
		if (h_ != TTM_INVALID_HANDLE && vt_.destroy != nullptr) {
			vt_.destroy(h_);
		}
	}

	float step(int64_t global_step) override {
		if (vt_.step == nullptr || h_ == TTM_INVALID_HANDLE) return 0.0f;
		return vt_.step(h_, global_step);
	}

private:
	ttm_scheduler_vtable vt_;
	ttm_handle           h_;
};

/// Load all plugins listed in the config into a fresh PluginManager.
static std::expected<ttm::plugins::PluginManager, std::string>
setup_plugins(const ttm::conf::TrainingConfig& cfg) {
	auto mgrResult = ttm::plugins::PluginManager::create();
	if (!mgrResult) return std::unexpected(mgrResult.error());

	for (const auto& entry : cfg.plugins) {
		if (auto r = mgrResult->load(entry.path, entry.config); !r) {
			return std::unexpected(
				"Failed to load plugin '" + entry.path + "': " + r.error()
			);
		}
	}
	return mgrResult;
}

/* =========================================================================
 * ttm fit
 * ====================================================================== */

static int cmd_fit(
	const std::vector<std::filesystem::path>& config_files,
	const std::vector<std::string>&           set_overrides
) {
	// 1. Load & merge config
	auto cfgResult = ttm::conf::load_config(
		std::span{config_files}, std::span{set_overrides}
	);
	if (!cfgResult) {
		std::cerr << "ttm fit: " << cfgResult.error() << '\n';
		return 1;
	}
	const auto& cfg = *cfgResult;

	// 2. Set up plugins
	auto mgrResult = setup_plugins(cfg);
	if (!mgrResult) {
		std::cerr << "ttm fit: " << mgrResult.error() << '\n';
		return 1;
	}
	auto& mgr = *mgrResult;

	// 3. Resolve dataset source
	const std::string scheme = scheme_of(cfg.dataset.uri);
	auto* source = mgr.find_source(scheme);
	if (source == nullptr) {
		std::cerr << "ttm fit: no plugin registered for dataset scheme '" << scheme << "'\n"
		          << "         Add the appropriate plugin to the 'plugins:' section.\n";
		return 1;
	}

	// 4. Dataset factory — invoked once per epoch for a fresh iterator
	const auto train_ds = cfg.dataset; // capture by value
	auto train_factory  = [source, train_ds]()
		-> std::expected<std::unique_ptr<ttm::datasets::DatasetIterator>, std::string>
	{
		return ttm::datasets::load_dataset(
			*source, train_ds.uri, train_ds.split, train_ds.config_name
		);
	};

	// 5. Optional validation factory
	ttm::trainer::DatasetFactory val_factory;
	if (cfg.validation) {
		const auto val_ds  = *cfg.validation;
		const auto val_uri = val_ds.uri.empty() ? cfg.dataset.uri : val_ds.uri;
		const auto val_cfg = val_ds.config_name.empty()
			? cfg.dataset.config_name : val_ds.config_name;

		auto* val_src = mgr.find_source(scheme_of(val_uri));
		if (val_src == nullptr) {
			std::cerr << "ttm fit: no plugin for validation dataset scheme '"
			          << scheme_of(val_uri) << "'\n";
			return 1;
		}
		val_factory = [val_src, val_uri, val_cfg, split = val_ds.split]()
			-> std::expected<std::unique_ptr<ttm::datasets::DatasetIterator>, std::string>
		{
			return ttm::datasets::load_dataset(*val_src, val_uri, split, val_cfg);
		};
	}

	// 6. Load model via plugin-registered loader
	const auto dev = ttm::model::Device::from_string(
		cfg.model.device + (cfg.model.device_id != 0
			? ":" + std::to_string(cfg.model.device_id)
			: "")
	);
	auto pipelineResult = ttm::model::ModelPipeline::load(
		cfg.model, cfg.preprocessors, mgr, dev
	);
	if (!pipelineResult) {
		std::cerr << "ttm fit: model load failed: " << pipelineResult.error() << '\n';
		return 1;
	}

	// 7. Build and run trainer
	auto trainer = ttm::trainer::Trainer(
		cfg, mgr, std::move(*pipelineResult), train_factory
	);
	if (val_factory) trainer.validation(std::move(val_factory));

	// 8. Attach LR scheduler if one is registered for the configured type
	if (const auto* vt = mgr.find_scheduler_vtable(cfg.scheduler.type)) {
		char sched_cfg[256];
		std::snprintf(sched_cfg, sizeof(sched_cfg),
			"{\"warmup_steps\":%lld,\"min_lr\":%f,\"step_size\":%lld,\"gamma\":%f,\"total_steps\":%lld}",
			static_cast<long long>(cfg.scheduler.warmup_steps),
			static_cast<double>(cfg.scheduler.min_lr),
			static_cast<long long>(cfg.scheduler.step_size),
			static_cast<double>(cfg.scheduler.gamma),
			static_cast<long long>(cfg.scheduler.total_steps)
		);
		const auto h = vt->create(cfg.optimizer.lr, sched_cfg,
		                          static_cast<uint32_t>(std::strlen(sched_cfg)));
		if (h != TTM_INVALID_HANDLE) {
			trainer.scheduler(std::make_unique<VtableScheduler>(*vt, h));
		}
	}

	auto result = trainer.fit();
	if (!result) {
		std::cerr << "ttm fit: training failed: " << result.error() << '\n';
		return 1;
	}
	return 0;
}

/* =========================================================================
 * ttm validate
 * ====================================================================== */

static int cmd_validate(
	const std::vector<std::filesystem::path>& config_files,
	const std::vector<std::string>&           set_overrides
) {
	auto cfgResult = ttm::conf::load_config(
		std::span{config_files}, std::span{set_overrides}
	);
	if (!cfgResult) {
		std::cerr << "ttm validate: " << cfgResult.error() << '\n';
		return 1;
	}
	const auto& cfg = *cfgResult;

	std::cout << "Config OK\n\n";
	std::cout << "  dataset:    " << cfg.dataset.uri;
	if (!cfg.dataset.config_name.empty())
		std::cout << "  [" << cfg.dataset.config_name << "]";
	std::cout << "  split=" << cfg.dataset.split
	          << "  batch_size=" << cfg.dataset.batch_size << '\n';
	if (cfg.validation) {
		const auto& v = *cfg.validation;
		std::cout << "  validation: "
		          << (v.uri.empty() ? "(same dataset)" : v.uri)
		          << "  split=" << v.split
		          << "  batch_size=" << v.batch_size << '\n';
	}
	std::cout << "  model:      "
	          << (cfg.model.path.empty() ? "(none)" : cfg.model.path)
	          << "  device=" << cfg.model.device << '\n';
	std::cout << "  optimizer:  " << cfg.optimizer.type
	          << "  lr=" << cfg.optimizer.lr
	          << "  wd=" << cfg.optimizer.weight_decay << '\n';
	std::cout << "  scheduler:  " << cfg.scheduler.type
	          << "  warmup=" << cfg.scheduler.warmup_steps << '\n';
	std::cout << "  training:   epochs=" << cfg.epochs
	          << "  grad_accum=" << cfg.gradient_accumulation_steps
	          << "  clip=" << cfg.grad_clip_norm << '\n';
	std::cout << "  plugins:    " << cfg.plugins.size() << '\n';
	for (const auto& p : cfg.plugins) std::cout << "    - " << p.path << '\n';
	return 0;
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(int argc, char** argv) {
	CLI::App app{"ttm — Train My Model", "ttm"};
	app.set_version_flag("--version", TTM_VERSION);
	app.require_subcommand(1);

	// ── ttm fit ────────────────────────────────────────────────────────────
	auto* fit = app.add_subcommand("fit", "Train a model from a YAML config");
	std::vector<std::filesystem::path> fit_configs;
	std::vector<std::string>           fit_set;
	fit->add_option("config", fit_configs,
		"One or more YAML config files (deep-merged left-to-right)")->required();
	fit->add_option("--set,-s", fit_set,
		"Override a config value, e.g. --set optimizer.lr=1e-4");
	fit->callback([&] { std::exit(cmd_fit(fit_configs, fit_set)); });

	// ── ttm validate ───────────────────────────────────────────────────────
	auto* validate = app.add_subcommand("validate", "Validate a config file without training");
	std::vector<std::filesystem::path> val_configs;
	std::vector<std::string>           val_set;
	validate->add_option("config", val_configs, "YAML config file(s)")->required();
	validate->add_option("--set,-s", val_set, "Override a config value");
	validate->callback([&] { std::exit(cmd_validate(val_configs, val_set)); });

	// ── ttm predict ────────────────────────────────────────────────────────
	app.add_subcommand("predict", "Run inference (not yet implemented)")
		->callback([&] {
			std::cerr << "ttm predict: not yet implemented\n";
			std::exit(2);
		});

	CLI11_PARSE(app, argc, argv);
	return 0;
}

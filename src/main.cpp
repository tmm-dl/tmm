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

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h> // GetModuleFileNameW
#else
#include <unistd.h> // readlink
#endif

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
		if (vt_.step == nullptr || h_ == TTM_INVALID_HANDLE)
			return 0.0f;
		return vt_.step(h_, global_step);
	}

private:
	ttm_scheduler_vtable vt_;
	ttm_handle h_;
};

/// Return the directory containing the running executable.
static std::filesystem::path exe_dir() {
#ifdef _WIN32
	wchar_t buf[4096];
	DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
	if (n == 0 || n == std::size(buf))
		return {};
	return std::filesystem::path(buf).parent_path();
#else
	char buf[4096];
	const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
	if (n <= 0)
		return {};
	buf[n] = '\0';
	return std::filesystem::path(buf).parent_path();
#endif
}

/// Return the platform-specific filename for a plugin named `name`.
/// E.g. "core" → "ttm_core.so" (Linux) / "ttm_core.dylib" (macOS) / "ttm_core.dll" (Windows)
static std::string plugin_filename(const std::string& name) {
#ifdef _WIN32
	return "ttm_" + name + ".dll";
#elif defined(__APPLE__)
	return "ttm_" + name + ".dylib";
#else
	return "ttm_" + name + ".so";
#endif
}

/// Resolve a plugin name to an absolute path.
/// Search order: exe dir, then each dir in TTM_PLUGIN_PATH (colon-separated).
/// Within each directory, also checks the `<name>/` subdirectory so that a
/// build tree like `build/extensions/core/ttm_core.so` is found when
/// TTM_PLUGIN_PATH=build/extensions.
static std::expected<std::filesystem::path, std::string> resolve_plugin_name(const std::string& name) {
	const std::string filename = plugin_filename(name);
	const std::string filenameWasm = "ttm_" + name + ".wasm";

	// Check `base/filename` and `base/<name>/filename` for both native and WASM.
	auto try_dir = [&](const std::filesystem::path& base) -> std::optional<std::filesystem::path> {
		for (const auto& fn : {filename, filenameWasm}) {
			if (auto c = base / fn; std::filesystem::exists(c))
				return c;
			if (auto c = base / name / fn; std::filesystem::exists(c))
				return c;
		}
		return std::nullopt;
	};

	// 1. Next to the executable
	const auto exd = exe_dir();
	if (!exd.empty()) {
		if (auto c = try_dir(exd))
			return *c;
	}

	// 2. TTM_PLUGIN_PATH environment variable
	// NOLINTNEXTLINE(concurrency-mt-unsafe)
	const char* env = std::getenv("TTM_PLUGIN_PATH");
	if (env != nullptr) {
		std::string dirs{env};
		std::size_t start = 0;
		while (start < dirs.size()) {
#ifdef _WIN32
			const char sep = ';';
#else
			const char sep = ':';
#endif
			const auto end = dirs.find(sep, start);
			const std::string dir = dirs.substr(start, end == std::string::npos ? end : end - start);
			start = (end == std::string::npos) ? dirs.size() : end + 1;
			if (!dir.empty()) {
				if (auto c = try_dir(std::filesystem::path(dir)))
					return *c;
			}
		}
	}

	return std::unexpected(
			"plugin '" + name + "' not found (looked for '" + filename + "' or '" + filenameWasm +
			"').\n"
			"  Place " +
			filename + " (or .wasm) next to the ttm binary, or set TTM_PLUGIN_PATH."
	);
}

/// Load all plugins listed in the config into a fresh PluginManager.
static std::expected<ttm::plugins::PluginManager, std::string> setup_plugins(const ttm::conf::TrainingConfig& cfg) {
	auto mgrResult = ttm::plugins::PluginManager::create();
	if (!mgrResult)
		return std::unexpected(mgrResult.error());

	for (const auto& entry : cfg.plugins) {
		// Resolve name → path when only a name is given
		std::string resolved_path = entry.path;
		if (resolved_path.empty() && !entry.name.empty()) {
			auto r = resolve_plugin_name(entry.name);
			if (!r) {
				if (entry.optional) {
					std::fprintf(
							stderr, "ttm: optional plugin '%s' skipped: %s\n", entry.name.c_str(), r.error().c_str()
					);
					continue;
				}
				return std::unexpected(r.error());
			}
			resolved_path = r->string();
		}
		if (resolved_path.empty()) {
			if (entry.optional)
				continue;
			return std::unexpected("Plugin entry has neither 'name' nor 'path' set.");
		}

		if (auto r = mgrResult->load(resolved_path, entry.config); !r) {
			if (entry.optional) {
				std::fprintf(
						stderr, "ttm: optional plugin '%s' skipped: %s\n", resolved_path.c_str(), r.error().c_str()
				);
				continue;
			}
			return std::unexpected("Failed to load plugin '" + resolved_path + "': " + r.error());
		}
	}
	return mgrResult;
}

/* =========================================================================
 * ttm fit
 * ====================================================================== */

/* =========================================================================
 * SIGINT handling
 * ====================================================================== */

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<bool> g_sigint{false};

#ifndef _WIN32
// Previous handler to chain (set before we install ours).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static struct sigaction g_prevSigint{};
#endif

/// SIGINT handler: set a flag for graceful shutdown, then forward to the
/// previous handler (typically Python's, which calls PyErr_SetInterrupt so
/// the next bytecode tick raises KeyboardInterrupt).
static void onSigint(int sig) {
	g_sigint.store(true, std::memory_order_relaxed);
#ifndef _WIN32
	// Chain to previous handler (Python's default SIGINT handler, or SIG_DFL).
	const auto& prev = g_prevSigint;
	if (prev.sa_handler != SIG_DFL && prev.sa_handler != SIG_IGN && prev.sa_handler != nullptr) {
		prev.sa_handler(sig);
	}
#else
	(void)sig;
#endif
}

static int
cmd_fit(const std::vector<std::filesystem::path>& config_files, const std::vector<std::string>& set_overrides) {
	// 1. Load & merge config
	auto cfgResult = ttm::conf::load_config(std::span{config_files}, std::span{set_overrides});
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
	auto train_factory = [source,
						  train_ds]() -> std::expected<std::unique_ptr<ttm::datasets::DatasetIterator>, std::string> {
		return ttm::datasets::load_dataset(
				*source, train_ds.uri, train_ds.split, train_ds.config_name, train_ds.batch_size
		);
	};

	// 5. Optional validation factory
	ttm::trainer::DatasetFactory val_factory;
	if (cfg.validation) {
		const auto val_ds = *cfg.validation;
		const auto val_uri = val_ds.uri.empty() ? cfg.dataset.uri : val_ds.uri;
		const auto val_cfg = val_ds.config_name.empty() ? cfg.dataset.config_name : val_ds.config_name;

		auto* val_src = mgr.find_source(scheme_of(val_uri));
		if (val_src == nullptr) {
			std::cerr << "ttm fit: no plugin for validation dataset scheme '" << scheme_of(val_uri) << "'\n";
			return 1;
		}
		const int64_t val_batch = val_ds.batch_size > 0 ? val_ds.batch_size : cfg.dataset.batch_size;
		val_factory = [val_src, val_uri, val_cfg, split = val_ds.split,
					   val_batch]() -> std::expected<std::unique_ptr<ttm::datasets::DatasetIterator>, std::string> {
			return ttm::datasets::load_dataset(*val_src, val_uri, split, val_cfg, val_batch);
		};
	}

	// 6. Load model via plugin-registered loader
	const auto dev = ttm::model::Device::from_string(
			cfg.model.device + (cfg.model.device_id != 0 ? ":" + std::to_string(cfg.model.device_id) : "")
	);
	auto pipelineResult = ttm::model::ModelPipeline::load(cfg.model, cfg.preprocessors, mgr, dev);
	if (!pipelineResult) {
		std::cerr << "ttm fit: model load failed: " << pipelineResult.error() << '\n';
		return 1;
	}

	// 7. Build and run trainer
	// Capture handle before move (model_handle() unavailable after std::move)
	const ttm_handle model_h = (*pipelineResult)->model_handle();
	auto trainer = ttm::trainer::Trainer(cfg, mgr, std::move(*pipelineResult), train_factory);
	if (val_factory)
		trainer.validation(std::move(val_factory));

	// 8. Create and attach the optimizer (required for parameter updates)
	if (!cfg.optimizer.type.empty()) {
		char opt_cfg[512];
		std::snprintf(
				opt_cfg, sizeof(opt_cfg),
				"{\"lr\":%g,\"weight_decay\":%g,\"beta1\":%g,\"beta2\":%g,"
				"\"eps\":%g,\"amsgrad\":%d,\"device\":\"%s\"}",
				static_cast<double>(cfg.optimizer.lr), static_cast<double>(cfg.optimizer.weight_decay),
				static_cast<double>(cfg.optimizer.beta1), static_cast<double>(cfg.optimizer.beta2),
				static_cast<double>(cfg.optimizer.eps), cfg.optimizer.amsgrad ? 1 : 0, cfg.model.device.c_str()
		);

		std::string opt_err;
		auto optimizer = mgr.make_optimizer(cfg.optimizer.type, model_h, nullptr, 0, nullptr, opt_cfg, &opt_err);
		if (optimizer) {
			trainer.optimizer(std::move(optimizer));
		} else {
			std::cerr << "ttm fit: optimizer '" << cfg.optimizer.type << "' unavailable: " << opt_err << "\n"
					  << "  Make sure the plugin providing this optimizer is loaded.\n";
			return 1;
		}
	}

	// 9. Attach LR scheduler if one is registered for the configured type
	if (const auto* vt = mgr.find_scheduler_vtable(cfg.scheduler.type)) {
		char sched_cfg[256];
		std::snprintf(
				sched_cfg, sizeof(sched_cfg),
				"{\"warmup_steps\":%lld,\"min_lr\":%f,\"step_size\":%lld,\"gamma\":%f,\"total_steps\":%lld}",
				static_cast<long long>(cfg.scheduler.warmup_steps), static_cast<double>(cfg.scheduler.min_lr),
				static_cast<long long>(cfg.scheduler.step_size), static_cast<double>(cfg.scheduler.gamma),
				static_cast<long long>(cfg.scheduler.total_steps)
		);
		const auto h = vt->create(cfg.optimizer.lr, sched_cfg, static_cast<uint32_t>(std::strlen(sched_cfg)));
		if (h != TTM_INVALID_HANDLE) {
			trainer.scheduler(std::make_unique<VtableScheduler>(*vt, h));
		}
	}

	// Install SIGINT handler for graceful interruption during training.
	g_sigint.store(false, std::memory_order_relaxed);
#ifndef _WIN32
	{
		struct sigaction sa{};
		sa.sa_handler = onSigint;
		sigemptyset(&sa.sa_mask);
		sa.sa_flags = 0;
		sigaction(SIGINT, &sa, &g_prevSigint);
	}
#else
	std::signal(SIGINT, onSigint);
#endif
	trainer.stopPredicate([] { return g_sigint.load(std::memory_order_relaxed); });

	auto result = trainer.fit();

	// Restore default SIGINT handler.
#ifndef _WIN32
	sigaction(SIGINT, &g_prevSigint, nullptr);
#else
	std::signal(SIGINT, SIG_DFL);
#endif

	if (!result) {
		std::cerr << "ttm fit: training failed: " << result.error() << '\n';
		return 1;
	}
	return 0;
}

/* =========================================================================
 * ttm validate
 * ====================================================================== */

static int
cmd_validate(const std::vector<std::filesystem::path>& config_files, const std::vector<std::string>& set_overrides) {
	auto cfgResult = ttm::conf::load_config(std::span{config_files}, std::span{set_overrides});
	if (!cfgResult) {
		std::cerr << "ttm validate: " << cfgResult.error() << '\n';
		return 1;
	}
	const auto& cfg = *cfgResult;

	std::cout << "Config OK\n\n";
	std::cout << "  dataset:    " << cfg.dataset.uri;
	if (!cfg.dataset.config_name.empty())
		std::cout << "  [" << cfg.dataset.config_name << "]";
	std::cout << "  split=" << cfg.dataset.split << "  batch_size=" << cfg.dataset.batch_size << '\n';
	if (cfg.validation) {
		const auto& v = *cfg.validation;
		std::cout << "  validation: " << (v.uri.empty() ? "(same dataset)" : v.uri) << "  split=" << v.split
				  << "  batch_size=" << v.batch_size << '\n';
	}
	std::cout << "  model:      " << (cfg.model.path.empty() ? "(none)" : cfg.model.path)
			  << "  device=" << cfg.model.device << '\n';
	std::cout << "  optimizer:  " << cfg.optimizer.type << "  lr=" << cfg.optimizer.lr
			  << "  wd=" << cfg.optimizer.weight_decay << '\n';
	std::cout << "  scheduler:  " << cfg.scheduler.type << "  warmup=" << cfg.scheduler.warmup_steps << '\n';
	std::cout << "  training:   epochs=" << cfg.epochs << "  grad_accum=" << cfg.gradient_accumulation_steps
			  << "  clip=" << cfg.grad_clip_norm << '\n';
	std::cout << "  plugins:    " << cfg.plugins.size() << '\n';
	for (const auto& p : cfg.plugins)
		std::cout << "    - " << p.path << '\n';
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
	std::vector<std::string> fit_set;
	fit->add_option("config", fit_configs, "One or more YAML config files (deep-merged left-to-right)")->required();
	fit->add_option("--set,-s", fit_set, "Override a config value, e.g. --set optimizer.lr=1e-4");
	fit->callback([&] { std::exit(cmd_fit(fit_configs, fit_set)); });

	// ── ttm validate ───────────────────────────────────────────────────────
	auto* validate = app.add_subcommand("validate", "Validate a config file without training");
	std::vector<std::filesystem::path> val_configs;
	std::vector<std::string> val_set;
	validate->add_option("config", val_configs, "YAML config file(s)")->required();
	validate->add_option("--set,-s", val_set, "Override a config value");
	validate->callback([&] { std::exit(cmd_validate(val_configs, val_set)); });

	// ── ttm predict ────────────────────────────────────────────────────────
	app.add_subcommand("predict", "Run inference (not yet implemented)")->callback([&] {
		std::cerr << "ttm predict: not yet implemented\n";
		std::exit(2);
	});

	CLI11_PARSE(app, argc, argv);
	return 0;
}

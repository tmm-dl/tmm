/**
 * @file main.cpp
 * @brief tmm — Train My Model — CLI entry point.
 *
 * ### Commands
 * - `tmm fit <config.yml> [config2.yml …] [--set key=value …]`
 *   Load config, set up plugins, and run the training loop.
 *
 * - `tmm validate <config.yml> [--set key=value …]`
 *   Parse and print the resolved config without training.
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

#include <tmm/compat/format.hpp>
#include <tmm/conf/config.hpp>
#include <tmm/conf/loader.hpp>
#include <tmm/datasets/dataset_loader.hpp>
#include <tmm/model/device.hpp>
#include <tmm/model/pipeline.hpp>
#include <tmm/plugins/plugin_manager.hpp>
#include <tmm/trainer/trainer.hpp>

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
static std::string schemeOf(const std::string& uri) {
	const auto colon = uri.find(':');
	return (colon != std::string::npos) ? uri.substr(0, colon + 1) : "file:";
}

/**
 * @brief ILRScheduler backed by a plugin-registered tmm_scheduler_vtable.
 *
 * Wraps the C vtable so that PluginManager::findSchedulerVtable() results
 * can be attached to a Trainer without exposing the C ABI above this file.
 */
class VtableScheduler final : public tmm::trainer::ILRScheduler {
public:
	VtableScheduler(const tmm_scheduler_vtable& vt, tmm_handle h) : vt_(vt), h_(h) {}

	~VtableScheduler() override {
		if (h_ != TMM_INVALID_HANDLE && vt_.destroy != nullptr) {
			vt_.destroy(h_);
		}
	}

	float step(int64_t globalStep) override {
		if (vt_.step == nullptr || h_ == TMM_INVALID_HANDLE)
			return 0.0f;
		return vt_.step(h_, globalStep);
	}

private:
	tmm_scheduler_vtable vt_;
	tmm_handle h_;
};

/// Return the directory containing the running executable.
static std::filesystem::path exeDir() {
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
/// E.g. "core" → "tmm_core.so" (Linux) / "tmm_core.dylib" (macOS) / "tmm_core.dll" (Windows)
static std::string pluginFilename(const std::string& name) {
#ifdef _WIN32
	return "tmm_" + name + ".dll";
#elif defined(__APPLE__)
	return "tmm_" + name + ".dylib";
#else
	return "tmm_" + name + ".so";
#endif
}

static std::vector<std::string> pluginFilenameCandidates(const std::string& name) {
	return {
#ifdef _WIN32
			std::format("tmm_{}.dll", name),
#elif defined(__APPLE__)
			std::format("tmm_{}.dylib", name),
#else
			std::format("tmm_{}.so", name),
#endif
			std::format("tmm_{}.wasm", name)
	};
}

/// Resolve a plugin name to an absolute path.
/// Search order: exe dir, then each dir in TMM_PLUGIN_PATH (colon-separated).
/// Within each directory, also checks the `<name>/` subdirectory so that a
/// build tree like `build/extensions/core/tmm_core.so` is found when
/// TMM_PLUGIN_PATH=build/extensions.
static std::expected<std::filesystem::path, std::string> resolvePluginName(const std::string& name) {
	const std::string filename = pluginFilename(name);
	const std::string filenameWasm = "tmm_" + name + ".wasm";

	// Check `base/filename` and `base/<name>/filename` for both native and WASM.
	auto try_dir = [&](const std::filesystem::path& base) -> std::optional<std::filesystem::path> {
		for (const auto& fn : pluginFilenameCandidates(name)) {
			if (auto c = base / fn; std::filesystem::exists(c))
				return c;
			if (auto c = base / name / fn; std::filesystem::exists(c))
				return c;
		}
		return std::nullopt;
	};

	// 1. Next to the executable
	const auto exd = exeDir();
	if (!exd.empty()) {
		if (auto c = try_dir(exd))
			return *c;
	}

	// 2. TMM_PLUGIN_PATH environment variable
	// NOLINTNEXTLINE(concurrency-mt-unsafe)
	const char* env = std::getenv("TMM_PLUGIN_PATH");
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

	constexpr auto msgFmt = "plugin '{}' not found (looked for '{}' or '{}').\n Place {} (or .wasm) next to the tmm "
							"binary, or set TMM_PLUGIN_PATH";
	return std::unexpected(std::format(msgFmt, name, filename, filenameWasm, filename));

	// return std::unexpected(
	// 		"plugin '" + name + "' not found (looked for '" + filename + "' or '" + filenameWasm +
	// 		"').\n"
	// 		"  Place " +
	// 		filename + " (or .wasm) next to the tmm binary, or set TMM_PLUGIN_PATH."
	// );
}

/// Load all plugins listed in the config into a fresh PluginManager.
static std::expected<tmm::plugins::PluginManager, std::string> setupPlugins(const tmm::conf::TrainingConfig& cfg) {
	auto mgrResult = tmm::plugins::PluginManager::create();
	if (!mgrResult)
		return std::unexpected(mgrResult.error());

	for (const auto& entry : cfg.plugins) {
		// Resolve name → path when only a name is given
		std::string resolvedPath = entry.path;
		if (resolvedPath.empty() && !entry.name.empty()) {
			auto r = resolvePluginName(entry.name);
			if (!r) {
				if (entry.optional) {
					std::fprintf(
							stderr, "tmm: optional plugin '%s' skipped: %s\n", entry.name.c_str(), r.error().c_str()
					);
					continue;
				}
				return std::unexpected(r.error());
			}
			resolvedPath = r->string();
		}
		if (resolvedPath.empty()) {
			if (entry.optional)
				continue;
			return std::unexpected("Plugin entry has neither 'name' nor 'path' set.");
		}

		if (auto r = mgrResult->load(resolvedPath, entry.config); !r) {
			if (entry.optional) {
				std::fprintf(
						stderr, "tmm: optional plugin '%s' skipped: %s\n", resolvedPath.c_str(), r.error().c_str()
				);
				continue;
			}
			return std::unexpected("Failed to load plugin '" + resolvedPath + "': " + r.error());
		}
	}
	return mgrResult;
}

/* =========================================================================
 * tmm fit
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

static int cmdFit(const std::vector<std::filesystem::path>& configFiles, const std::vector<std::string>& setOverrides) {
	// 1. Load & merge config
	auto cfgResult = tmm::conf::load_config(std::span{configFiles}, std::span{setOverrides});
	if (!cfgResult) {
		std::cerr << "tmm fit: " << cfgResult.error() << '\n';
		return 1;
	}
	const auto& cfg = *cfgResult;

	// 2. Set up plugins
	auto mgrResult = setupPlugins(cfg);
	if (!mgrResult) {
		std::cerr << "tmm fit: " << mgrResult.error() << '\n';
		return 1;
	}
	auto& mgr = *mgrResult;

	// 3. Resolve dataset source
	const std::string scheme = schemeOf(cfg.dataset.uri);
	auto* source = mgr.findSource(scheme);
	if (source == nullptr) {
		std::cerr << "tmm fit: no plugin registered for dataset scheme '" << scheme << "'\n"
				  << "         Add the appropriate plugin to the 'plugins:' section.\n";
		return 1;
	}

	// 4. Dataset factory — invoked once per epoch for a fresh iterator
	const auto trainDs = cfg.dataset; // capture by value
	auto trainFactory = [source,
						 trainDs]() -> std::expected<std::unique_ptr<tmm::datasets::DatasetIterator>, std::string> {
		return tmm::datasets::load_dataset(*source, trainDs.uri, trainDs.split, trainDs.configName, trainDs.batchSize);
	};

	// 5. Optional validation factory
	tmm::trainer::DatasetFactory valFactory;
	if (cfg.validation) {
		const auto valDs = *cfg.validation;
		const auto valUri = valDs.uri.empty() ? cfg.dataset.uri : valDs.uri;
		const auto valCfg = valDs.configName.empty() ? cfg.dataset.configName : valDs.configName;

		auto* valSrc = mgr.findSource(schemeOf(valUri));
		if (valSrc == nullptr) {
			std::cerr << "tmm fit: no plugin for validation dataset scheme '" << schemeOf(valUri) << "'\n";
			return 1;
		}
		const int64_t valBatch = valDs.batchSize > 0 ? valDs.batchSize : cfg.dataset.batchSize;
		valFactory = [valSrc, valUri, valCfg, split = valDs.split,
					  valBatch]() -> std::expected<std::unique_ptr<tmm::datasets::DatasetIterator>, std::string> {
			return tmm::datasets::load_dataset(*valSrc, valUri, split, valCfg, valBatch);
		};
	}

	// 6. Load model via plugin-registered loader
	const auto dev = tmm::model::Device::from_string(
			cfg.model.device + (cfg.model.deviceId != 0 ? ":" + std::to_string(cfg.model.deviceId) : "")
	);
	auto pipelineResult = tmm::model::ModelPipeline::load(cfg.model, cfg.preprocessors, mgr, dev);
	if (!pipelineResult) {
		std::cerr << "tmm fit: model load failed: " << pipelineResult.error() << '\n';
		return 1;
	}

	// 7. Build and run trainer
	// Capture handle before move (model_handle() unavailable after std::move)
	const tmm_handle modelH = (*pipelineResult)->model_handle();
	auto trainer = tmm::trainer::Trainer(cfg, mgr, std::move(*pipelineResult), trainFactory);
	if (valFactory)
		trainer.validation(std::move(valFactory));

	// 8. Create and attach the optimizer (required for parameter updates)
	if (!cfg.optimizer.type.empty()) {
		char optCfg[512];
		std::snprintf(
				optCfg, sizeof(optCfg),
				"{\"lr\":%g,\"weight_decay\":%g,\"beta1\":%g,\"beta2\":%g,"
				"\"eps\":%g,\"amsgrad\":%d,\"device\":\"%s\"}",
				static_cast<double>(cfg.optimizer.lr), static_cast<double>(cfg.optimizer.weightDecay),
				static_cast<double>(cfg.optimizer.beta1), static_cast<double>(cfg.optimizer.beta2),
				static_cast<double>(cfg.optimizer.eps), cfg.optimizer.amsgrad ? 1 : 0, cfg.model.device.c_str()
		);

		std::string optErr;
		auto optimizer = mgr.makeOptimizer(cfg.optimizer.type, modelH, nullptr, 0, nullptr, optCfg, &optErr);
		if (optimizer) {
			trainer.optimizer(std::move(optimizer));
		} else {
			std::cerr << "tmm fit: optimizer '" << cfg.optimizer.type << "' unavailable: " << optErr << "\n"
					  << "  Make sure the plugin providing this optimizer is loaded.\n";
			return 1;
		}
	}

	// 9. Attach LR scheduler if one is registered for the configured type
	if (const auto* vt = mgr.findSchedulerVtable(cfg.scheduler.type)) {
		char schedCfg[256];
		std::snprintf(
				schedCfg, sizeof(schedCfg),
				"{\"warmup_steps\":%lld,\"min_lr\":%f,\"step_size\":%lld,\"gamma\":%f,\"total_steps\":%lld}",
				static_cast<long long>(cfg.scheduler.warmupSteps), static_cast<double>(cfg.scheduler.minLr),
				static_cast<long long>(cfg.scheduler.stepSize), static_cast<double>(cfg.scheduler.gamma),
				static_cast<long long>(cfg.scheduler.totalSteps)
		);
		const auto h = vt->create(cfg.optimizer.lr, schedCfg, static_cast<uint32_t>(std::strlen(schedCfg)));
		if (h != TMM_INVALID_HANDLE) {
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
		std::cerr << "tmm fit: training failed: " << result.error() << '\n';
		return 1;
	}
	return 0;
}

/* =========================================================================
 * tmm validate
 * ====================================================================== */

static int
cmdValidate(const std::vector<std::filesystem::path>& configFiles, const std::vector<std::string>& setOverrides) {
	auto cfgResult = tmm::conf::load_config(std::span{configFiles}, std::span{setOverrides});
	if (!cfgResult) {
		std::cerr << "tmm validate: " << cfgResult.error() << '\n';
		return 1;
	}
	const auto& cfg = *cfgResult;

	std::cout << "Config OK\n\n";
	std::cout << "  dataset:    " << cfg.dataset.uri;
	if (!cfg.dataset.configName.empty())
		std::cout << "  [" << cfg.dataset.configName << "]";
	std::cout << "  split=" << cfg.dataset.split << "  batch_size=" << cfg.dataset.batchSize << '\n';
	if (cfg.validation) {
		const auto& v = *cfg.validation;
		std::cout << "  validation: " << (v.uri.empty() ? "(same dataset)" : v.uri) << "  split=" << v.split
				  << "  batch_size=" << v.batchSize << '\n';
	}
	std::cout << "  model:      " << (cfg.model.path.empty() ? "(none)" : cfg.model.path)
			  << "  device=" << cfg.model.device << '\n';
	std::cout << "  optimizer:  " << cfg.optimizer.type << "  lr=" << cfg.optimizer.lr
			  << "  wd=" << cfg.optimizer.weightDecay << '\n';
	std::cout << "  scheduler:  " << cfg.scheduler.type << "  warmup=" << cfg.scheduler.warmupSteps << '\n';
	std::cout << "  training:   epochs=" << cfg.epochs << "  grad_accum=" << cfg.gradientAccumulationSteps
			  << "  clip=" << cfg.gradClipNorm << '\n';
	std::cout << "  plugins:    " << cfg.plugins.size() << '\n';
	for (const auto& p : cfg.plugins)
		std::cout << "    - " << p.path << '\n';
	return 0;
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(int argc, char** argv) {
	CLI::App app{"tmm — Train My Model", "tmm"};
	app.set_version_flag("--version", TMM_VERSION);
	app.require_subcommand(1);

	// ── tmm fit ────────────────────────────────────────────────────────────
	auto* fit = app.add_subcommand("fit", "Train a model from a YAML config");
	std::vector<std::filesystem::path> fitConfigs;
	std::vector<std::string> fitSet;
	fit->add_option("config", fitConfigs, "One or more YAML config files (deep-merged left-to-right)")->required();
	fit->add_option("--set,-s", fitSet, "Override a config value, e.g. --set optimizer.lr=1e-4");
	fit->callback([&] { std::exit(cmdFit(fitConfigs, fitSet)); });

	// ── tmm validate ───────────────────────────────────────────────────────
	auto* validate = app.add_subcommand("validate", "Validate a config file without training");
	std::vector<std::filesystem::path> valConfigs;
	std::vector<std::string> valSet;
	validate->add_option("config", valConfigs, "YAML config file(s)")->required();
	validate->add_option("--set,-s", valSet, "Override a config value");
	validate->callback([&] { std::exit(cmdValidate(valConfigs, valSet)); });

	CLI11_PARSE(app, argc, argv);
	return 0;
}

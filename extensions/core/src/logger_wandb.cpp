/**
 * @file logger_wandb.cpp
 * @brief Weights & Biases (wandb) callback — writes JSONL metrics files.
 *
 * @details
 * On fit begin, creates a timestamped run directory under `logDir` and opens
 * a `wandb-history.jsonl` file.  Each epoch appends one JSON line with all
 * metrics plus `_step` and `_timestamp`.  On fit end the file is flushed and,
 * if the wandb CLI is on PATH and `sync` is true, `wandb sync <runDir>` is
 * spawned as a subprocess so the run appears in the wandb web UI.
 *
 * JSON config fields:
 *   project  (str)  — wandb project name (default: "tmm")
 *   name     (str)  — run display name (default: auto-generated)
 *   logDir   (str)  — base directory for run dirs (default: "wandb")
 *   sync     (bool) — attempt `wandb sync` on fit end (default: true)
 *
 * Also implements loggersRegister() and loggersTeardown() for both loggers.
 */

#include "loggers.hpp"

#include <tmm/plugins/abi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <string_view>

namespace {

	/* ============================================================================
	 * Minimal JSON helpers
	 * ========================================================================== */

	static std::string jsonGetStr(std::string_view json, std::string_view key, std::string_view def = {}) {
		const std::string needle = std::string("\"") + std::string(key) + "\":";
		const auto pos = json.find(needle);
		if (pos == std::string_view::npos)
			return std::string(def);
		auto val = json.substr(pos + needle.size());
		while (!val.empty() && (val.front() == ' ' || val.front() == '\t'))
			val.remove_prefix(1);
		if (val.empty())
			return std::string(def);
		if (val.front() == '"') {
			val.remove_prefix(1);
			const auto end = val.find('"');
			return end == std::string_view::npos ? std::string(def) : std::string(val.substr(0, end));
		}
		const auto end = val.find_first_of(",}] \t\n");
		return std::string(val.substr(0, end == std::string_view::npos ? val.size() : end));
	}

	static bool jsonGetBool(std::string_view json, std::string_view key, bool def) {
		const auto s = jsonGetStr(json, key);
		if (s == "true")
			return true;
		if (s == "false")
			return false;
		return def;
	}

	/// Append all key-value pairs from a metrics JSON object to an output string,
	/// omitting non-numeric values.  Writes: `"key":value,` pairs (no trailing comma).
	static std::string metricsToJsonPairs(std::string_view json) {
		// Re-emit the full json body as-is, stripping the outer braces so we can
		// inject _step and _timestamp alongside the metrics.
		auto start = json.find('{');
		auto end = json.rfind('}');
		if (start == std::string_view::npos || end == std::string_view::npos)
			return {};
		// Return inner content (may have trailing whitespace/commas — caller handles)
		auto inner = std::string(json.substr(start + 1, end - start - 1));
		// Trim trailing whitespace/commas
		while (!inner.empty() &&
			   (inner.back() == ' ' || inner.back() == '\t' || inner.back() == '\n' || inner.back() == ','))
			inner.pop_back();
		return inner;
	}

	/* ============================================================================
	 * Slot allocator
	 * ========================================================================== */

	constexpr int kMaxWandbSlots = 64;

	struct WandbState {
		char project[128] = "tmm";
		char name[128] = "";
		char logDir[512] = "wandb";
		char runDir[640] = "";
		bool doSync = true;
		FILE* histFile = nullptr;
		int64_t step = 0;
	};

	static WandbState s_wb_slots[kMaxWandbSlots];
	static bool s_wb_used[kMaxWandbSlots] = {};

	static tmm_handle wbAlloc() {
		for (int i = 0; i < kMaxWandbSlots; ++i) {
			if (!s_wb_used[i]) {
				s_wb_used[i] = true;
				s_wb_slots[i] = WandbState{};
				return static_cast<tmm_handle>(i);
			}
		}
		return TMM_INVALID_HANDLE;
	}

	static WandbState* wbGet(tmm_handle h) {
		if (h < 0 || h >= kMaxWandbSlots || !s_wb_used[static_cast<int>(h)])
			return nullptr;
		return &s_wb_slots[static_cast<int>(h)];
	}

	/* ============================================================================
	 * Callback implementation
	 * ========================================================================== */

	static tmm_handle wandbCreate(const char* cfg, uint32_t cfgLen) {
		const auto h = wbAlloc();
		if (h == TMM_INVALID_HANDLE)
			return TMM_INVALID_HANDLE;
		auto* st = wbGet(h);
		const std::string_view json{cfg, cfgLen};
		const auto project = jsonGetStr(json, "project", "tmm");
		const auto name = jsonGetStr(json, "name", "");
		const auto logDir = jsonGetStr(json, "logDir", "wandb");
		st->doSync = jsonGetBool(json, "sync", true);
		std::strncpy(st->project, project.c_str(), sizeof(st->project) - 1);
		std::strncpy(st->name, name.c_str(), sizeof(st->name) - 1);
		std::strncpy(st->logDir, logDir.c_str(), sizeof(st->logDir) - 1);
		return h;
	}

	static void wandbOnFitBegin(tmm_handle h, const char* /*metrics*/, uint32_t /*len*/) {
		auto* st = wbGet(h);
		if (st == nullptr)
			return;
		st->step = 0;

		// Build run directory: <logDir>/run_<timestamp>
		const auto ts = static_cast<long long>(std::time(nullptr));
		const std::string runDir = std::string(st->logDir) + "/run_" + std::to_string(ts);
		std::strncpy(st->runDir, runDir.c_str(), sizeof(st->runDir) - 1);

		std::error_code ec;
		std::filesystem::create_directories(runDir, ec);

		// Write wandb-metadata.json
		const std::string metaPath = runDir + "/wandb-metadata.json";
		if (FILE* mf = std::fopen(metaPath.c_str(), "w"); mf != nullptr) {
			std::fprintf(mf, "{\"project\":\"%s\",\"name\":\"%s\",\"startedAt\":%lld}\n", st->project, st->name, ts);
			std::fclose(mf);
		}

		// Open history file
		const std::string histPath = runDir + "/wandb-history.jsonl";
		st->histFile = std::fopen(histPath.c_str(), "w");
	}

	static int32_t wandbOnEpochEnd(tmm_handle h, int64_t epoch, const char* metricsJson, uint32_t len) {
		auto* st = wbGet(h);
		if (st == nullptr || st->histFile == nullptr)
			return 0;
		st->step = epoch + 1;

		const double ts = static_cast<double>(std::time(nullptr));
		const auto pairs = metricsToJsonPairs({metricsJson, len});

		if (pairs.empty()) {
			std::fprintf(st->histFile, "{\"_step\":%lld,\"_timestamp\":%.3f}\n", static_cast<long long>(st->step), ts);
		} else {
			std::fprintf(
					st->histFile, "{%s,\"_step\":%lld,\"_timestamp\":%.3f}\n", pairs.c_str(),
					static_cast<long long>(st->step), ts
			);
		}
		std::fflush(st->histFile);
		return 0;
	}

	static void wandbOnFitEnd(tmm_handle h, const char* metricsJson, uint32_t len) {
		auto* st = wbGet(h);
		if (st == nullptr)
			return;

		// Write final summary
		if (st->histFile != nullptr) {
			const double ts = static_cast<double>(std::time(nullptr));
			const auto pairs = metricsToJsonPairs({metricsJson, len});
			if (!pairs.empty()) {
				std::fprintf(
						st->histFile, "{%s,\"_step\":%lld,\"_timestamp\":%.3f}\n", pairs.c_str(),
						static_cast<long long>(st->step), ts
				);
			}
			std::fclose(st->histFile);
			st->histFile = nullptr;
		}

		// Write wandb-summary.json
		const std::string sumPath = std::string(st->runDir) + "/wandb-summary.json";
		if (FILE* sf = std::fopen(sumPath.c_str(), "w"); sf != nullptr) {
			const auto pairs = metricsToJsonPairs({metricsJson, len});
			if (pairs.empty()) {
				std::fprintf(sf, "{}\n");
			} else {
				std::fprintf(sf, "{%s}\n", pairs.c_str());
			}
			std::fclose(sf);
		}

		// Attempt `wandb sync <runDir>` if requested
		if (st->doSync && st->runDir[0] != '\0') {
			std::string cmd = std::string("wandb sync \"") + st->runDir + "\" 2>/dev/null";
			std::system(cmd.c_str()); // best-effort; ignored if wandb not installed // NOLINT
		}
	}

	static void wandbDestroy(tmm_handle h) {
		auto* st = wbGet(h);
		if (st != nullptr && st->histFile != nullptr) {
			std::fclose(st->histFile);
			st->histFile = nullptr;
		}
		if (h >= 0 && h < kMaxWandbSlots)
			s_wb_used[static_cast<int>(h)] = false;
	}

} // anonymous namespace

tmm_trainer_callback_vtable g_cb_wandb = {
		/* create         */ wandbCreate,
		/* on_fit_begin   */ wandbOnFitBegin,
		/* on_epoch_begin */ nullptr,
		/* on_epoch_end   */ wandbOnEpochEnd,
		/* on_fit_end     */ wandbOnFitEnd,
		/* destroy        */ wandbDestroy,
};

void wandbTeardownSlots() {
	for (int i = 0; i < kMaxWandbSlots; ++i) {
		if (s_wb_used[i] && s_wb_slots[i].histFile != nullptr) {
			std::fclose(s_wb_slots[i].histFile);
			s_wb_slots[i].histFile = nullptr;
		}
		s_wb_used[i] = false;
	}
}

/* ============================================================================
 * Combined registration / teardown for both loggers
 * ========================================================================== */

void loggersRegister(const tmm_host_api* host) {
	if (host == nullptr || host->register_callback == nullptr)
		return;
	host->register_callback(host->ctx, "tensorBoard", &g_cb_tensorBoard);
	host->register_callback(host->ctx, "wandb", &g_cb_wandb);
}

void loggersTeardown() {
	tbTeardownSlots();
	wandbTeardownSlots();
}

/**
 * @file callbacks.cpp
 * @brief Built-in trainer callbacks: early_stopping and checkpoint.
 *
 * @details
 * Both callbacks are implemented as plain C state structs wrapped in
 * ttm_trainer_callback_vtable entries.  JSON config is parsed with a minimal
 * hand-rolled parser that avoids external dependencies.
 */

#include "callbacks.hpp"

#include <ttm/plugins/abi.h>

#include <array>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>

namespace {

	/* ============================================================================
	 * Minimal JSON helpers
	 * ========================================================================== */

	/// Extract a string value for `key` from flat JSON, or return `def`.
	static std::string json_get_str(std::string_view json, std::string_view key, std::string_view def = {}) {
		// Find "key":
		const std::string needle = std::string("\"") + std::string(key) + "\":";
		const auto pos = json.find(needle);
		if (pos == std::string_view::npos)
			return std::string(def);
		auto val = json.substr(pos + needle.size());
		// Skip whitespace
		while (!val.empty() && (val.front() == ' ' || val.front() == '\t'))
			val.remove_prefix(1);
		if (val.empty())
			return std::string(def);
		if (val.front() == '"') {
			val.remove_prefix(1);
			const auto end = val.find('"');
			return end == std::string_view::npos ? std::string(def) : std::string(val.substr(0, end));
		}
		// Number or keyword: read until delimiter
		const auto end = val.find_first_of(",}] \t\n");
		return std::string(val.substr(0, end == std::string_view::npos ? val.size() : end));
	}

	static int64_t json_get_int(std::string_view json, std::string_view key, int64_t def) {
		const auto s = json_get_str(json, key);
		if (s.empty())
			return def;
		try {
			return std::stoll(s);
		} catch (...) {
			return def;
		}
	}

	static float json_get_float(std::string_view json, std::string_view key, float def) {
		const auto s = json_get_str(json, key);
		if (s.empty())
			return def;
		try {
			return std::stof(s);
		} catch (...) {
			return def;
		}
	}

	/// Find a named float metric in a metrics JSON object, or return NaN.
	static float metrics_get(std::string_view metrics_json, std::string_view key) {
		const auto s = json_get_str(metrics_json, key);
		if (s.empty() || s == "null")
			return std::numeric_limits<float>::quiet_NaN();
		try {
			return std::stof(s);
		} catch (...) {
			return std::numeric_limits<float>::quiet_NaN();
		}
	}

	/* ============================================================================
	 * Slot allocator — up to 64 concurrent instances per callback type
	 * ========================================================================== */

	constexpr int kMaxSlots = 64;

	/* ============================================================================
	 * EarlyStopping
	 * ========================================================================== */

	struct EarlyStoppingState {
		char monitor[128] = "val_loss";
		int32_t patience = 5;
		bool mode_min = true; ///< true = minimize, false = maximize
		float min_delta = 0.0f;
		float best = std::numeric_limits<float>::infinity();
		int32_t wait = 0;
		bool stopped = false;
	};

	static EarlyStoppingState s_es_slots[kMaxSlots];
	static bool s_es_used[kMaxSlots] = {};

	static ttm_handle es_alloc() {
		for (int i = 0; i < kMaxSlots; ++i) {
			if (!s_es_used[i]) {
				s_es_used[i] = true;
				s_es_slots[i] = EarlyStoppingState{};
				return static_cast<ttm_handle>(i);
			}
		}
		return TTM_INVALID_HANDLE;
	}

	static EarlyStoppingState* es_get(ttm_handle h) {
		if (h < 0 || h >= kMaxSlots || !s_es_used[static_cast<int>(h)])
			return nullptr;
		return &s_es_slots[static_cast<int>(h)];
	}

	static ttm_handle early_stopping_create(const char* cfg, uint32_t cfg_len) {
		const auto h = es_alloc();
		if (h == TTM_INVALID_HANDLE)
			return TTM_INVALID_HANDLE;
		auto* st = es_get(h);
		const std::string_view json{cfg, cfg_len};
		const auto monitor = json_get_str(json, "monitor", "val_loss");
		const auto mode = json_get_str(json, "mode", "min");
		std::strncpy(st->monitor, monitor.c_str(), sizeof(st->monitor) - 1);
		st->patience = static_cast<int32_t>(json_get_int(json, "patience", 5));
		st->mode_min = (mode != "max");
		st->min_delta = json_get_float(json, "min_delta", 0.0f);
		return h;
	}

	static void early_stopping_on_fit_begin(ttm_handle h, const char* /*metrics*/, uint32_t /*len*/) {
		auto* st = es_get(h);
		if (st == nullptr)
			return;
		st->best = st->mode_min ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
		st->wait = 0;
		st->stopped = false;
	}

	static int32_t
	early_stopping_on_epoch_end(ttm_handle h, int64_t /*epoch*/, const char* metrics_json, uint32_t len) {
		auto* st = es_get(h);
		if (st == nullptr || st->stopped)
			return (st != nullptr && st->stopped) ? 1 : 0;

		const std::string_view mj{metrics_json, len};
		const float current = metrics_get(mj, st->monitor);
		if (std::isnan(current))
			return 0; // metric not yet available — don't stop

		const bool improved =
				st->mode_min ? (current < st->best - st->min_delta) : (current > st->best + st->min_delta);

		if (improved) {
			st->best = current;
			st->wait = 0;
		} else {
			++st->wait;
			if (st->wait >= st->patience) {
				st->stopped = true;
				return 1; // request early stopping
			}
		}
		return 0;
	}

	static void early_stopping_destroy(ttm_handle h) {
		if (h >= 0 && h < kMaxSlots)
			s_es_used[static_cast<int>(h)] = false;
	}

} // anonymous namespace

ttm_trainer_callback_vtable g_cb_early_stopping = {
		/* create         */ early_stopping_create,
		/* on_fit_begin   */ early_stopping_on_fit_begin,
		/* on_epoch_begin */ nullptr,
		/* on_epoch_end   */ early_stopping_on_epoch_end,
		/* on_fit_end     */ nullptr,
		/* destroy        */ early_stopping_destroy,
};

namespace {

	/* ============================================================================
	 * Checkpoint
	 * ========================================================================== */

	struct CheckpointState {
		char directory[512] = "checkpoints";
		int32_t every_n_epochs = 1;
		int32_t epoch_count = 0;
	};

	static CheckpointState s_ck_slots[kMaxSlots];
	static bool s_ck_used[kMaxSlots] = {};

	static ttm_handle ck_alloc() {
		for (int i = 0; i < kMaxSlots; ++i) {
			if (!s_ck_used[i]) {
				s_ck_used[i] = true;
				s_ck_slots[i] = CheckpointState{};
				return static_cast<ttm_handle>(i);
			}
		}
		return TTM_INVALID_HANDLE;
	}

	static CheckpointState* ck_get(ttm_handle h) {
		if (h < 0 || h >= kMaxSlots || !s_ck_used[static_cast<int>(h)])
			return nullptr;
		return &s_ck_slots[static_cast<int>(h)];
	}

	static ttm_handle checkpoint_create(const char* cfg, uint32_t cfg_len) {
		const auto h = ck_alloc();
		if (h == TTM_INVALID_HANDLE)
			return TTM_INVALID_HANDLE;
		auto* st = ck_get(h);
		const std::string_view json{cfg, cfg_len};
		// Support both "directory" and "dir" keys
		auto dir = json_get_str(json, "directory");
		if (dir.empty())
			dir = json_get_str(json, "dir", "checkpoints");
		std::strncpy(st->directory, dir.c_str(), sizeof(st->directory) - 1);
		// Support both "every_n_epochs" and "save_every_n_epochs"
		auto n = json_get_int(json, "every_n_epochs", -1);
		if (n < 0)
			n = json_get_int(json, "save_every_n_epochs", 1);
		st->every_n_epochs = static_cast<int32_t>(n);
		return h;
	}

	static void checkpoint_on_fit_begin(ttm_handle h, const char* /*metrics*/, uint32_t /*len*/) {
		auto* st = ck_get(h);
		if (st == nullptr)
			return;
		st->epoch_count = 0;
		// Ensure the checkpoint directory exists
		std::error_code ec;
		std::filesystem::create_directories(st->directory, ec);
	}

	static int32_t checkpoint_on_epoch_end(ttm_handle h, int64_t epoch, const char* metrics_json, uint32_t len) {
		auto* st = ck_get(h);
		if (st == nullptr)
			return 0;
		++st->epoch_count;
		if (st->epoch_count % st->every_n_epochs != 0)
			return 0;

		// Write a JSON marker file: checkpoints/epoch_{N}.json
		const std::string path = std::string(st->directory) + "/epoch_" + std::to_string(epoch + 1) + ".json";
		if (std::ofstream f{path}; f) {
			f << "{\"epoch\":" << (epoch + 1) << ",\"metrics\":" << std::string_view{metrics_json, len} << "}\n";
		}
		return 0;
	}

	static void checkpoint_destroy(ttm_handle h) {
		if (h >= 0 && h < kMaxSlots)
			s_ck_used[static_cast<int>(h)] = false;
	}

} // anonymous namespace

ttm_trainer_callback_vtable g_cb_checkpoint = {
		/* create         */ checkpoint_create,
		/* on_fit_begin   */ checkpoint_on_fit_begin,
		/* on_epoch_begin */ nullptr,
		/* on_epoch_end   */ checkpoint_on_epoch_end,
		/* on_fit_end     */ nullptr,
		/* destroy        */ checkpoint_destroy,
};

/* ============================================================================
 * Public registration / teardown
 * ========================================================================== */

void callbacksRegister(const ttm_host_api* host) {
	if (host == nullptr || host->register_callback == nullptr)
		return;
	host->register_callback(host->ctx, "early_stopping", &g_cb_early_stopping);
	host->register_callback(host->ctx, "checkpoint", &g_cb_checkpoint);
}

void callbacksTeardown() {
	// Reset all slot tables so they're clean if the plugin is reloaded
	for (int i = 0; i < kMaxSlots; ++i) {
		s_es_used[i] = false;
		s_ck_used[i] = false;
	}
}

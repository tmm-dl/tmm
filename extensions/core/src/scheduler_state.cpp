/**
 * @file scheduler_state.cpp
 * @brief Scheduler state table, allocation helpers, and config parsing.
 */

#include "scheduler_state.hpp"

#include <cstdio>
#include <string>
#include <string_view>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
SchedulerState g_scheds[kMaxScheds]{};

ttm_handle allocSched(SchedulerState s) {
	for (int i = 0; i < kMaxScheds; ++i) {
		if (!g_scheds[i].used) {
			s.used = true;
			g_scheds[i] = s;
			return static_cast<ttm_handle>(i);
		}
	}
	return TTM_INVALID_HANDLE;
}

SchedulerState* getSched(ttm_handle h) {
	if (h < 0 || h >= kMaxScheds) return nullptr;
	return g_scheds[static_cast<int>(h)].used ? &g_scheds[static_cast<int>(h)] : nullptr;
}

void freeSched(ttm_handle h) {
	if (h >= 0 && h < kMaxScheds) g_scheds[static_cast<int>(h)] = {};
}

void schedDestroy(ttm_handle h) { freeSched(h); }

/* -------------------------------------------------------------------------
 * Minimal JSON scalar extraction (no external library dependency)
 * ---------------------------------------------------------------------- */

namespace {

float jsonFloat(const char* json, uint32_t len, const char* key, float def) {
	if (json == nullptr || len == 0) return def;
	const std::string_view j{json, len};
	const std::string search = std::string("\"") + key + "\"";
	auto pos = j.find(search);
	if (pos == std::string_view::npos) return def;
	pos = j.find(':', pos + search.size());
	if (pos == std::string_view::npos) return def;
	++pos;
	while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) { ++pos; }
	float val = def;
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- sscanf requires raw pointer; bounded by 'len'
	std::sscanf(j.data() + pos, "%f", &val);
	return val;
}

int64_t jsonInt64(const char* json, uint32_t len, const char* key, int64_t def) {
	if (json == nullptr || len == 0) return def;
	const std::string_view j{json, len};
	const std::string search = std::string("\"") + key + "\"";
	auto pos = j.find(search);
	if (pos == std::string_view::npos) return def;
	pos = j.find(':', pos + search.size());
	if (pos == std::string_view::npos) return def;
	++pos;
	while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) { ++pos; }
	long long val = static_cast<long long>(def);
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- see jsonFloat
	std::sscanf(j.data() + pos, "%lld", &val);
	return static_cast<int64_t>(val);
}

} // anonymous namespace

SchedulerState parseSchedCfg(float base_lr, const char* cfg, uint32_t cfg_len) {
	SchedulerState s;
	s.base_lr      = base_lr;
	s.warmup_steps = jsonInt64(cfg, cfg_len, "warmup_steps", 0);
	s.min_lr       = jsonFloat(cfg, cfg_len, "min_lr",       0.0f);
	s.step_size    = jsonInt64(cfg, cfg_len, "step_size",    1);
	s.gamma        = jsonFloat(cfg, cfg_len, "gamma",        0.1f);
	s.total_steps  = jsonInt64(cfg, cfg_len, "total_steps",  0);
	return s;
}

float warmupFactor(const SchedulerState& s, int64_t step) {
	if (s.warmup_steps <= 0 || step >= s.warmup_steps) return 1.0f;
	return static_cast<float>(step) / static_cast<float>(s.warmup_steps);
}

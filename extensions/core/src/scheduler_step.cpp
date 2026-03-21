/**
 * @file scheduler_step.cpp
 * @brief Step LR scheduler — multiplicative decay every N steps.
 *
 * lr = base_lr * gamma^floor(global_step / step_size), floored at min_lr.
 */

#include "scheduler_state.hpp"

#include <ttm/plugins/abi.h>

#include <algorithm>
#include <cstdint>

namespace {

ttm_handle stepCreate(float base_lr, const char* cfg, uint32_t cfg_len) {
	return allocSched(parseSchedCfg(base_lr, cfg, cfg_len));
}

float stepStep(ttm_handle h, int64_t global_step) {
	const auto* s = getSched(h);
	if (!s) return 0.0f;
	const int64_t n = (s->step_size > 0) ? (global_step / s->step_size) : 0;
	float lr = s->base_lr;
	for (int64_t i = 0; i < n; ++i) { lr *= s->gamma; }
	return std::max(lr, s->min_lr);
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ttm_scheduler_vtable g_sched_step = { stepCreate, stepStep, schedDestroy };

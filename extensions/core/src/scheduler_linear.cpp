/**
 * @file scheduler_linear.cpp
 * @brief Linear LR scheduler — optional warmup then linear decay to min_lr.
 */

#include "scheduler_state.hpp"

#include <ttm/plugins/abi.h>

#include <algorithm>
#include <cstdint>

namespace {

ttm_handle linearCreate(float base_lr, const char* cfg, uint32_t cfg_len) {
	return allocSched(parseSchedCfg(base_lr, cfg, cfg_len));
}

float linearStep(ttm_handle h, int64_t global_step) {
	const auto* s = getSched(h);
	if (!s) return 0.0f;
	if (global_step < s->warmup_steps) {
		return s->base_lr * warmupFactor(*s, global_step);
	}
	const int64_t decay_steps = s->total_steps - s->warmup_steps;
	const int64_t decay_step  = global_step - s->warmup_steps;
	if (decay_steps > 0 && decay_step >= 0) {
		const float progress = std::min(
			static_cast<float>(decay_step) / static_cast<float>(decay_steps), 1.0f);
		return std::max(s->base_lr * (1.0f - progress), s->min_lr);
	}
	return s->base_lr;
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ttm_scheduler_vtable g_sched_linear = { linearCreate, linearStep, schedDestroy };

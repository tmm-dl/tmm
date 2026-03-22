/**
 * @file scheduler_cosine_warmup.cpp
 * @brief Cosine annealing LR scheduler with linear warmup.
 *
 * Phase 1: linear ramp from 0 → base_lr over warmup_steps.
 * Phase 2: cosine decay from base_lr → min_lr over (total_steps - warmup_steps).
 *
 * Returns base_lr if total_steps == warmup_steps (no decay range).
 */

#include "scheduler_state.hpp"

#include <ttm/plugins/abi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

	ttm_handle cosineWarmupCreate(float base_lr, const char* cfg, uint32_t cfg_len) {
		return allocSched(parseSchedCfg(base_lr, cfg, cfg_len));
	}

	float cosineWarmupStep(ttm_handle h, int64_t global_step) {
		const auto* s = getSched(h);
		if (!s)
			return 0.0f;
		if (global_step < s->warmup_steps) {
			return s->base_lr * warmupFactor(*s, global_step);
		}
		const int64_t decay_steps = s->total_steps - s->warmup_steps;
		const int64_t decay_step = global_step - s->warmup_steps;
		if (decay_steps <= 0)
			return s->base_lr;
		const float progress = std::min(static_cast<float>(decay_step) / static_cast<float>(decay_steps), 1.0f);
		return s->min_lr + 0.5f * (s->base_lr - s->min_lr) * (1.0f + std::cos(kPi * progress));
	}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ttm_scheduler_vtable g_sched_cosine_warmup = {cosineWarmupCreate, cosineWarmupStep, schedDestroy};

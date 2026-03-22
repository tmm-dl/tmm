/**
 * @file scheduler_cosine.cpp
 * @brief Cosine annealing LR scheduler (no warmup).
 *
 * lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * step / total_steps))
 *
 * Returns base_lr unchanged when total_steps == 0 (no decay limit).
 */

#include "scheduler_state.hpp"

#include <tmm/plugins/abi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

	tmm_handle cosineCreate(float base_lr, const char* cfg, uint32_t cfg_len) {
		return allocSched(parseSchedCfg(base_lr, cfg, cfg_len));
	}

	float cosineStep(tmm_handle h, int64_t global_step) {
		const auto* s = getSched(h);
		if (!s)
			return 0.0f;
		if (s->total_steps <= 0)
			return s->base_lr;
		const float progress = std::min(static_cast<float>(global_step) / static_cast<float>(s->total_steps), 1.0f);
		return s->min_lr + 0.5f * (s->base_lr - s->min_lr) * (1.0f + std::cos(kPi * progress));
	}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
tmm_scheduler_vtable g_sched_cosine = {cosineCreate, cosineStep, schedDestroy};

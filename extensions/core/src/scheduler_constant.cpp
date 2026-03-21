/**
 * @file scheduler_constant.cpp
 * @brief Constant LR scheduler — returns the base learning rate unchanged.
 */

#include "scheduler_state.hpp"

#include <ttm/plugins/abi.h>

namespace {

ttm_handle constantCreate(float base_lr, const char* cfg, uint32_t cfg_len) {
	return allocSched(parseSchedCfg(base_lr, cfg, cfg_len));
}

float constantStep(ttm_handle h, [[maybe_unused]] int64_t global_step) {
	const auto* s = getSched(h);
	return s ? s->base_lr : 0.0f;
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ttm_scheduler_vtable g_sched_constant = { constantCreate, constantStep, schedDestroy };

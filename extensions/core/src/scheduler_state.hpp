/**
 * @file scheduler_state.hpp
 * @brief Shared state types and helpers for built-in LR schedulers.
 *
 * @details
 * All five built-in schedulers (constant, step, linear, cosine, cosine_warmup)
 * share a flat state table `g_scheds[kMaxScheds]`.  Each scheduler handle is
 * an index into this table.
 */

#pragma once

#include <tmm/plugins/abi.h>

#include <cstdint>

inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr int kMaxScheds = 16;

struct SchedulerState {
	float base_lr = 1e-3f;
	int64_t warmup_steps = 0;
	float min_lr = 0.0f;
	int64_t step_size = 1;
	float gamma = 0.1f;
	int64_t total_steps = 0;
	bool used = false;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) -- plugin-level state table
extern SchedulerState g_scheds[kMaxScheds];

/** @brief Allocate a scheduler slot and return its handle. */
tmm_handle allocSched(SchedulerState s);

/** @brief Return a pointer to the scheduler state for handle @p h, or nullptr. */
SchedulerState* getSched(tmm_handle h);

/** @brief Free the scheduler slot for handle @p h. */
void freeSched(tmm_handle h);

/** @brief Vtable-compatible destroy callback — calls freeSched(). */
void schedDestroy(tmm_handle h);

/** @brief Parse scheduler config from the JSON blob passed to create(). */
SchedulerState parseSchedCfg(float base_lr, const char* cfg, uint32_t cfg_len);

/** @brief Linear warmup factor: ramps from 0 → 1 over warmup_steps. */
float warmupFactor(const SchedulerState& s, int64_t step);

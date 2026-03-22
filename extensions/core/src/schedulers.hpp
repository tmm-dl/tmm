/**
 * @file schedulers.hpp
 * @brief Built-in LR scheduler vtable declarations and lifecycle.
 */

#pragma once

#include <ttm/plugins/abi.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern ttm_scheduler_vtable g_sched_constant;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern ttm_scheduler_vtable g_sched_step;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern ttm_scheduler_vtable g_sched_linear;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern ttm_scheduler_vtable g_sched_cosine;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern ttm_scheduler_vtable g_sched_cosine_warmup;

/**
 * @brief Register all five built-in schedulers with the host.
 *
 * Registers: constant, step, linear, cosine, cosine_warmup. No-op if host->register_scheduler is nullptr.
 */
void schedulersRegister(const ttm_host_api* host);

/**
 * @brief Reset all scheduler slots (called on plugin teardown).
 */
void schedulersTeardown();

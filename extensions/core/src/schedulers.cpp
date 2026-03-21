/**
 * @file schedulers.cpp
 * @brief Registration and teardown for all built-in LR schedulers.
 */

#include "schedulers.hpp"
#include "scheduler_state.hpp"

void schedulersRegister(const ttm_host_api* host) {
	if (host->register_scheduler == nullptr) return;
	host->register_scheduler(host->ctx, "constant",      &g_sched_constant);
	host->register_scheduler(host->ctx, "step",          &g_sched_step);
	host->register_scheduler(host->ctx, "linear",        &g_sched_linear);
	host->register_scheduler(host->ctx, "cosine",        &g_sched_cosine);
	host->register_scheduler(host->ctx, "cosine_warmup", &g_sched_cosine_warmup);
}

void schedulersTeardown() {
	for (auto& s : g_scheds) { s = {}; }
}

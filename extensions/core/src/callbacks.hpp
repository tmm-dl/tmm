/**
 * @file callbacks.hpp
 * @brief Built-in trainer callback vtable declarations and lifecycle.
 *
 * @details
 * Provides two built-in callbacks registered with the host via
 * callbacksRegister():
 *
 *   - `early_stopping` — stops training when a monitored metric stops improving.
 *     JSON config fields: monitor (str), patience (int), mode ("min"/"max"),
 *     min_delta (float).
 *
 *   - `checkpoint` — saves a checkpoint marker file every N epochs.
 *     JSON config fields: directory (str), every_n_epochs (int).
 */

#pragma once

#include <tmm/plugins/abi.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern tmm_trainer_callback_vtable g_cb_early_stopping;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern tmm_trainer_callback_vtable g_cb_checkpoint;

/**
 * @brief Register all built-in callbacks with the host.
 * Registers: early_stopping, checkpoint.
 * No-op if host->register_callback is nullptr.
 */
void callbacksRegister(const tmm_host_api* host);

/**
 * @brief Reset all callback slots (called on plugin teardown).
 */
void callbacksTeardown();

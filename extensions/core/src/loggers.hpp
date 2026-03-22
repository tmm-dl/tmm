/**
 * @file loggers.hpp
 * @brief Built-in training logger callback vtable declarations and lifecycle.
 *
 * @details
 * Provides two logging callbacks registered with the host via loggersRegister():
 *
 *   - `tensorBoard` — writes TFEvents binary files for TensorBoard visualisation.
 *     JSON config fields: logDir (str, default "runs"), flushSecs (int, default 120).
 *
 *   - `wandb` — writes metrics as JSONL and optionally syncs via the wandb CLI.
 *     JSON config fields: project (str), name (str), logDir (str, default "wandb"),
 *     sync (bool, default true).
 */

#pragma once

#include <tmm/plugins/abi.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern tmm_trainer_callback_vtable g_cb_tensorBoard;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern tmm_trainer_callback_vtable g_cb_wandb;

/**
 * @brief Register all built-in logger callbacks with the host.
 * Registers: tensorBoard, wandb.
 * No-op if host->register_callback is nullptr.
 */
void loggersRegister(const tmm_host_api* host);

/**
 * @brief Reset all logger slots (called on plugin teardown).
 */
void loggersTeardown();

// Internal slot teardown helpers — implemented in each logger translation unit.
void tbTeardownSlots();
void wandbTeardownSlots();

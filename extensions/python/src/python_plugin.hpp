/**
 * @file python_plugin.hpp
 * @brief Internal header shared between python_model.cpp and python_optimizer.cpp.
 *
 * @details
 * Declares the cross-translation-unit helpers needed so the optimizer
 * implementation can access model objects, and so the plugin entry-point
 * (python_model.cpp) can delegate optimizer registration to the separate
 * python_optimizer.cpp translation unit.
 */

#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <ttm/plugins/abi.h>

/**
 * @brief Return the torch.nn.Module PyObject* for a model handle.
 *
 * @details
 * The Python plugin uses a flat slot table for loaded models.  This function
 * maps an opaque model handle (returned by the model loader vtable's load()
 * function) back to the underlying Python model object so that the optimizer
 * can call model.parameters() on it.
 *
 * @param h  Model handle returned by py_load().
 * @return Borrowed reference to the torch.nn.Module instance, or nullptr if
 *         @p h is invalid or the slot is unused.
 */
PyObject* py_get_model_obj(ttm_handle h);

/**
 * @brief Register all optimizer vtables (currently: adamw) with the host.
 *
 * @details
 * Called from ttm_plugin_init in python_model.cpp.  Registers:
 *   - "adamw"  — AdamW via torch.optim.AdamW
 *
 * @param host  Host API provided during plugin init.
 * @return #TTM_OK on success.
 */
ttm_error pyOptimizerRegister(const ttm_host_api* host);

/**
 * @brief Release all optimizer slots (called from ttm_plugin_teardown).
 */
void pyOptimizerTeardown();

/**
 * @brief Register the "hf-tokenize" transform vtable with the host.
 *
 * @param host  Host API provided during plugin init.
 * @return #TTM_OK on success.
 */
ttm_error pyTransformRegister(const ttm_host_api* host);

/**
 * @brief Release all transform slots (called from ttm_plugin_teardown).
 */
void pyTransformTeardown();

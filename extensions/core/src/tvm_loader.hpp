/**
 * @file tvm_loader.hpp
 * @brief TVM model loader — registration.
 */

#pragma once

#include <tmm/plugins/abi.h>

/**
 * @brief Register the TVM model loader with the host (if host supports it).
 *
 * Registers a stub loader when TMM_ENABLE_TVM=OFF (the default) that
 * probes .so/.tar files and returns a clear build-instruction error.
 */
void tvmLoaderRegister(const tmm_host_api* host);

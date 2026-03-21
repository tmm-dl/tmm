/**
 * @file tvm_loader.hpp
 * @brief TVM model loader — registration.
 */

#pragma once

#include <ttm/plugins/abi.h>

/**
 * @brief Register the TVM model loader with the host (if host supports it).
 *
 * Registers a stub loader when TTM_ENABLE_TVM=OFF (the default) that
 * probes .so/.tar files and returns a clear build-instruction error.
 */
void tvmLoaderRegister(const ttm_host_api* host);

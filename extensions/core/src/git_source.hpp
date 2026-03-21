/**
 * @file git_source.hpp
 * @brief Git-based dataset source — registration and lifecycle.
 */

#pragma once

#include <ttm/plugins/abi.h>

/**
 * @brief Initialise libgit2 and register the git dataset source with the host.
 * @return TTM_OK on success; error code on failure.
 */
ttm_error gitSourceRegister(const ttm_host_api* host);

/**
 * @brief Close all open file handles and shut down libgit2.
 */
void gitSourceTeardown();

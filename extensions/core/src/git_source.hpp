/**
 * @file git_source.hpp
 * @brief Git-based dataset source — registration and lifecycle.
 */

#pragma once

#include <tmm/plugins/abi.h>

/**
 * @brief Initialise libgit2 and register the git dataset source with the host.
 * @return TMM_OK on success; error code on failure.
 */
tmm_error gitSourceRegister(const tmm_host_api& host);

/**
 * @brief Close all open file handles and shut down libgit2.
 */
void gitSourceTeardown();

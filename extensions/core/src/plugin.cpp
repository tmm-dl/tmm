/**
 * @file plugin.cpp
 * @brief TTM core plugin — exported entry points.
 *
 * @details
 * Registers the following capabilities with the host on init:
 *   - Git-based dataset sources (gh: gl: bb: hf: sr:) via gitSourceRegister()
 *   - TVM model loader (stub unless TTM_ENABLE_TVM=ON) via tvmLoaderRegister()
 *   - Five built-in LR schedulers via schedulersRegister():
 *       constant, step, linear, cosine, cosine_warmup
 */

#include "callbacks.hpp"
#include "git_source.hpp"
#include "loggers.hpp"
#include "schedulers.hpp"
#include "tvm_loader.hpp"

#include <ttm/plugins/abi.h>
#include <ttm_core_export.h>

#include <cstdint>

static_assert(sizeof(ttm_handle) == sizeof(int64_t), "ttm_handle must be 64 bits");

namespace {

	ttm_plugin_info g_info = {TTM_ABI_VERSION, "core", "0.1.0", "Dataset sources: gh: gl: bb: hf: sr:"};

} // anonymous namespace

extern "C" {

TTM_CORE_EXPORT ttm_plugin_info* ttm_plugin_get_info(void) { return &g_info; }

TTM_CORE_EXPORT ttm_error ttm_plugin_init(
		const ttm_host_api* host, const char* /*cfg*/, uint32_t /*len*/
) {
	const ttm_error src_err = gitSourceRegister(host);
	if (src_err != TTM_OK)
		return src_err;

	tvmLoaderRegister(host);
	schedulersRegister(host);
	callbacksRegister(host);
	loggersRegister(host);

	return TTM_OK;
}

TTM_CORE_EXPORT void ttm_plugin_teardown(void) {
	gitSourceTeardown();
	schedulersTeardown();
	callbacksTeardown();
	loggersTeardown();
}

} // extern "C"

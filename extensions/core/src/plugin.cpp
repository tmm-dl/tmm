/**
 * @file plugin.cpp
 * @brief TMM core plugin — exported entry points.
 *
 * @details
 * Registers the following capabilities with the host on init:
 *   - Git-based dataset sources (gh: gl: bb: hf: sr:) via gitSourceRegister()
 *   - TVM model loader (stub unless TMM_ENABLE_TVM=ON) via tvmLoaderRegister()
 *   - Five built-in LR schedulers via schedulersRegister():
 *       constant, step, linear, cosine, cosine_warmup
 */

#include "callbacks.hpp"
#include "git_source.hpp"
#include "loggers.hpp"
#include "schedulers.hpp"
#include "tvm_loader.hpp"

#include <tmm/plugins/abi.h>
#include <tmm_core_export.h>

#include <cstdint>

static_assert(sizeof(tmm_handle) == sizeof(int64_t), "tmm_handle must be 64 bits");

namespace {

	tmm_plugin_info g_info = {TMM_ABI_VERSION, "core", "0.1.0", "Dataset sources: gh: gl: bb: hf: sr:"};

} // anonymous namespace

extern "C" {

TMM_CORE_EXPORT tmm_plugin_info* tmm_plugin_get_info(void) { return &g_info; }

TMM_CORE_EXPORT tmm_error tmm_plugin_init(
		const tmm_host_api* host, const char* /*cfg*/, uint32_t /*len*/
) {
	const tmm_error src_err = gitSourceRegister(host);
	if (src_err != TMM_OK)
		return src_err;

	tvmLoaderRegister(host);
	schedulersRegister(host);
	callbacksRegister(host);
	loggersRegister(host);

	return TMM_OK;
}

TMM_CORE_EXPORT void tmm_plugin_teardown(void) {
	gitSourceTeardown();
	schedulersTeardown();
	callbacksTeardown();
	loggersTeardown();
}

} // extern "C"

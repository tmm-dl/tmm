/**
 * @file tvm_loader.cpp
 * @brief TVM model loader implementation (stub unless TTM_ENABLE_TVM=ON).
 *
 * @details
 * When TTM_ENABLE_TVM=ON: full TVM runtime loader using tvm/runtime/c_runtime_api.h.
 * When TTM_ENABLE_TVM=OFF (default): probe-only stub that instructs the user
 * to rebuild with TVM enabled.
 */

#include "tvm_loader.hpp"

#include <ttm/plugins/abi.h>

#include <cstdio>
#include <string_view>

namespace {

#ifdef TTM_ENABLE_TVM

// Full TVM loader — requires the TVM runtime to be linked.
// Enabled via: cmake -DTTM_ENABLE_TVM=ON ...
// TODO: Replace with real TVM C runtime calls (tvm/runtime/c_runtime_api.h)

int32_t tvmProbe(const char* path, uint32_t len) {
	const std::string_view p{path, len};
	return (p.ends_with(".so") || p.ends_with(".tar")) ? 1 : 0;
}

ttm_handle tvmLoad(
	const char* /*path*/, uint32_t /*path_len*/,
	const char* /*cfg*/,  uint32_t /*cfg_len*/,
	char* err, uint32_t err_cap
) {
	std::snprintf(err, err_cap, "TVM loader: not yet fully implemented (TTM_ENABLE_TVM=ON)");
	return TTM_INVALID_HANDLE;
}

ttm_model_info_t tvmGetInfo(ttm_handle /*h*/) { return ttm_model_info_t{}; }

ttm_error tvmDescribeParams(ttm_handle /*h*/,
                             const ttm_param_desc_t** out_descs,
                             uint32_t* out_count) {
	if (out_descs) *out_descs = nullptr;
	if (out_count) *out_count = 0;
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmBindParams(ttm_handle /*h*/,
                         const DLTensor* /*params*/, uint32_t /*param_count*/,
                         const DLTensor* /*grads*/,  uint32_t /*grad_count*/) {
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmInitParams(ttm_handle /*h*/, const char* /*method*/, uint32_t /*len*/) {
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmStep(ttm_handle /*h*/,
                   const DLTensor* /*inputs*/, uint32_t /*n*/,
                   float* out_loss) {
	if (out_loss) *out_loss = 0.0f;
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmInfer(ttm_handle /*h*/,
                    const DLTensor* /*inputs*/,  uint32_t /*in_count*/,
                    DLTensor*       /*outputs*/, uint32_t* out_count) {
	if (out_count) *out_count = 0;
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmZeroGrad(ttm_handle /*h*/) { return TTM_OK; }
void      tvmDestroy(ttm_handle /*h*/)  {}

#else // TTM_ENABLE_TVM

/**
 * @brief Stub TVM loader registered when TTM_ENABLE_TVM=OFF (the default).
 *
 * Probes .so and .tar files so the host does not report "no loader found";
 * instead it returns a clear error instructing the user to rebuild with TVM.
 */
int32_t tvmProbe(const char* path, uint32_t len) {
	const std::string_view p{path, len};
	return (p.ends_with(".so") || p.ends_with(".tar")) ? 1 : 0;
}

ttm_handle tvmLoad(
	const char* /*path*/, uint32_t /*path_len*/,
	const char* /*cfg*/,  uint32_t /*cfg_len*/,
	char* err, uint32_t err_cap
) {
	std::snprintf(err, err_cap,
		"TVM model loader is not enabled in this build.\n"
		"Rebuild with: cmake -DTTM_ENABLE_TVM=ON"
	);
	return TTM_INVALID_HANDLE;
}

ttm_model_info_t tvmGetInfo(ttm_handle /*h*/) { return ttm_model_info_t{}; }

ttm_error tvmDescribeParams(ttm_handle /*h*/,
                             const ttm_param_desc_t** out, uint32_t* cnt) {
	if (out) *out = nullptr;
	if (cnt) *cnt = 0;
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmBindParams(ttm_handle /*h*/,
                         const DLTensor* /*p*/, uint32_t /*pc*/,
                         const DLTensor* /*g*/, uint32_t /*gc*/) {
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmInitParams(ttm_handle /*h*/, const char* /*m*/, uint32_t /*l*/) {
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmStep(ttm_handle /*h*/,
                   const DLTensor* /*in*/, uint32_t /*n*/,
                   float* loss) {
	if (loss) *loss = 0.0f;
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmInfer(ttm_handle /*h*/,
                    const DLTensor* /*in*/, uint32_t /*ic*/,
                    DLTensor* /*out*/, uint32_t* oc) {
	if (oc) *oc = 0;
	return TTM_ERR_UNSUPPORTED;
}

ttm_error tvmZeroGrad(ttm_handle /*h*/) { return TTM_OK; }
void      tvmDestroy(ttm_handle /*h*/)  {}

#endif // TTM_ENABLE_TVM

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ttm_model_loader_vtable g_tvm_loader = {
	tvmProbe, tvmLoad, tvmGetInfo,
	tvmDescribeParams, tvmBindParams, tvmInitParams,
	tvmStep, tvmInfer, tvmZeroGrad, tvmDestroy
};

} // anonymous namespace

void tvmLoaderRegister(const ttm_host_api* host) {
	if (host->register_model_loader != nullptr) {
		host->register_model_loader(host->ctx, &g_tvm_loader);
	}
}

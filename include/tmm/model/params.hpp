/**
 * @file params.hpp
 * @brief Model parameter buffer descriptor.
 * @ingroup tmm_model
 */

#pragma once

#include <dlpack/dlpack.h>

#include <cstdint>
#include <string>

namespace tmm::model {

	/**
	 * @brief Holds a named parameter tensor and its associated gradient buffer.
	 *
	 * @details
	 * Both `tensor` and `grad` are DLTensors whose `data` pointers are owned
	 * by the host (Trainer/ModelPipeline), not by the plugin.  The plugin
	 * receives these buffers via `tmm_model_loader_vtable::bind_params()` and
	 * must read/write through the provided pointers without freeing them.
	 *
	 * Memory layout:
	 * - CPU devices: aligned `std::malloc` / `posix_memalign`.
	 * - CUDA devices: `cudaMalloc` (future).
	 *
	 * @ingroup tmm_model
	 */
	struct ParamBuffer {
		std::string name;	   ///< Parameter name, e.g. `"transformer.h.0.attn.weight"`.
		DLTensor tensor{};	   ///< Parameter values tensor (host-allocated, device-resident).
		DLTensor grad{};	   ///< Gradient tensor (same shape/dtype as tensor, zeroed initially).
		bool trainable = true; ///< If false, zero_grad() skips this buffer and the optimizer ignores it.
	};

} // namespace tmm::model

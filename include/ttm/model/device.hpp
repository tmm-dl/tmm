/**
 * @file device.hpp
 * @brief Device descriptor for model execution.
 * @ingroup ttm_model
 */

#pragma once

#include <dlpack/dlpack.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace ttm::model {

	/**
	 * @brief Identifies the hardware device on which a model runs.
	 *
	 * @details
	 * Device strings follow the convention `"<type>"` or `"<type>:<id>"`,
	 * e.g. `"cpu"`, `"cuda:0"`, `"metal:0"`.  The `id` defaults to 0 when
	 * omitted.  from_string() is lenient: unknown strings fall back to CPU.
	 *
	 * @ingroup ttm_model
	 */
	struct Device {
		DLDeviceType type = kDLCPU; ///< DLPack device type constant.
		int32_t id = 0;				///< Device ordinal (GPU index, etc.).

		/**
		 * @brief Parse a device string into a Device.
		 *
		 * Supported strings:
		 * | String         | type       | id  |
		 * |----------------|------------|-----|
		 * | `"cpu"`        | kDLCPU     |  0  |
		 * | `"cuda"`       | kDLCUDA    |  0  |
		 * | `"cuda:N"`     | kDLCUDA    |  N  |
		 * | `"metal"`      | kDLMetal   |  0  |
		 * | `"metal:N"`    | kDLMetal   |  N  |
		 * | `"opencl"`     | kDLOpenCL  |  0  |
		 * | `"opencl:N"`   | kDLOpenCL  |  N  |
		 * | anything else  | kDLCPU     |  0  |
		 *
		 * @param s  Device string (e.g. `"cuda:0"`).
		 * @return Parsed device; falls back to CPU on unrecognised input.
		 */
		[[nodiscard]] static Device from_string(std::string_view s);

		/**
		 * @brief Serialise back to a human-readable device string.
		 * @return A string of the form `"cpu"`, `"cuda:0"`, etc.
		 */
		[[nodiscard]] std::string to_string() const;

		/** @brief Convert to a DLDevice (used by DLTensor). */
		[[nodiscard]] DLDevice to_dl() const noexcept { return {type, id}; }
	};

} // namespace ttm::model

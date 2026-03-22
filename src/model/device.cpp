/**
 * @file device.cpp
 * @brief Device::from_string / to_string implementation.
 */

#include <ttm/model/device.hpp>

#include <charconv>
#include <string>

namespace ttm::model {

	Device Device::from_string(std::string_view s) {
		// Find optional ":<id>" suffix
		const auto colon = s.rfind(':');
		std::string_view type_part = (colon == std::string_view::npos) ? s : s.substr(0, colon);
		int32_t id = 0;
		if (colon != std::string_view::npos) {
			const auto id_part = s.substr(colon + 1);
			std::from_chars(id_part.data(), id_part.data() + id_part.size(), id);
		}

		DLDeviceType type = kDLCPU;
		if (type_part == "cpu") {
			type = kDLCPU;
		} else if (type_part == "cuda") {
			type = kDLCUDA;
		} else if (type_part == "metal") {
			type = kDLMetal;
		} else if (type_part == "opencl") {
			type = kDLOpenCL;
		} else if (type_part == "vulkan") {
			type = kDLVulkan;
		} else if (type_part == "rocm") {
			type = kDLROCM;
		}
		// Unknown strings fall back to CPU (type already kDLCPU)

		return {type, id};
	}

	std::string Device::to_string() const {
		const char* name = "cpu";
		switch (type) {
		case kDLCPU:
			name = "cpu";
			break;
		case kDLCUDA:
			name = "cuda";
			break;
		case kDLMetal:
			name = "metal";
			break;
		case kDLOpenCL:
			name = "opencl";
			break;
		case kDLVulkan:
			name = "vulkan";
			break;
		case kDLROCM:
			name = "rocm";
			break;
		default:
			name = "cpu";
			break;
		}
		if (id == 0) {
			return name;
		}
		return std::string(name) + ':' + std::to_string(id);
	}

} // namespace ttm::model

/**
 * @file test_device.cpp
 * @brief Unit tests for ttm::model::Device — parsing and serialisation.
 */

#include <ttm/model/device.hpp>

#include <catch2/catch_test_macros.hpp>

using ttm::model::Device;

// =============================================================================
// from_string — well-known device types
// =============================================================================

TEST_CASE("Device::from_string parses 'cpu'", "[model][device]") {
	const auto d = Device::from_string("cpu");
	CHECK(d.type == kDLCPU);
	CHECK(d.id == 0);
}

TEST_CASE("Device::from_string parses 'cuda' without id", "[model][device]") {
	const auto d = Device::from_string("cuda");
	CHECK(d.type == kDLCUDA);
	CHECK(d.id == 0);
}

TEST_CASE("Device::from_string parses 'cuda:0'", "[model][device]") {
	const auto d = Device::from_string("cuda:0");
	CHECK(d.type == kDLCUDA);
	CHECK(d.id == 0);
}

TEST_CASE("Device::from_string parses 'cuda:3'", "[model][device]") {
	const auto d = Device::from_string("cuda:3");
	CHECK(d.type == kDLCUDA);
	CHECK(d.id == 3);
}

TEST_CASE("Device::from_string parses 'metal'", "[model][device]") {
	const auto d = Device::from_string("metal");
	CHECK(d.type == kDLMetal);
	CHECK(d.id == 0);
}

TEST_CASE("Device::from_string parses 'metal:1'", "[model][device]") {
	const auto d = Device::from_string("metal:1");
	CHECK(d.type == kDLMetal);
	CHECK(d.id == 1);
}

TEST_CASE("Device::from_string parses 'opencl'", "[model][device]") {
	const auto d = Device::from_string("opencl");
	CHECK(d.type == kDLOpenCL);
	CHECK(d.id == 0);
}

TEST_CASE("Device::from_string parses 'opencl:2'", "[model][device]") {
	const auto d = Device::from_string("opencl:2");
	CHECK(d.type == kDLOpenCL);
	CHECK(d.id == 2);
}

// =============================================================================
// from_string — fallback for unknown strings
// =============================================================================

TEST_CASE("Device::from_string falls back to CPU for empty string", "[model][device]") {
	const auto d = Device::from_string("");
	CHECK(d.type == kDLCPU);
	CHECK(d.id == 0);
}

TEST_CASE("Device::from_string falls back to CPU for unknown string", "[model][device]") {
	const auto d = Device::from_string("tpu:7");
	CHECK(d.type == kDLCPU);
}

// =============================================================================
// to_string — serialisation
// =============================================================================

TEST_CASE("Device::to_string serialises CPU device", "[model][device]") {
	Device d;
	d.type = kDLCPU;
	d.id = 0;
	CHECK(d.to_string() == "cpu");
}

TEST_CASE("Device::to_string serialises CUDA device with id", "[model][device]") {
	Device d;
	d.type = kDLCUDA;
	d.id = 2;
	CHECK(d.to_string() == "cuda:2");
}

TEST_CASE("Device::to_string serialises Metal device with id 0", "[model][device]") {
	// id == 0 is omitted from the serialised form (same as 'cpu' not 'cpu:0').
	Device d;
	d.type = kDLMetal;
	d.id = 0;
	CHECK(d.to_string() == "metal");
}

// =============================================================================
// Round-trip: from_string → to_string
// =============================================================================

TEST_CASE("Device round-trips 'cpu' through from_string/to_string", "[model][device]") {
	CHECK(Device::from_string("cpu").to_string() == "cpu");
}

TEST_CASE("Device round-trips 'cuda:1' through from_string/to_string", "[model][device]") {
	CHECK(Device::from_string("cuda:1").to_string() == "cuda:1");
}

// =============================================================================
// to_dl() helper
// =============================================================================

TEST_CASE("Device::to_dl returns matching DLDevice", "[model][device]") {
	Device d;
	d.type = kDLCUDA;
	d.id = 1;
	const DLDevice dl = d.to_dl();
	CHECK(dl.device_type == kDLCUDA);
	CHECK(dl.device_id == 1);
}

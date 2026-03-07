/**
 * @file test_plugin_manager.cpp
 * @brief Unit tests for PluginManager — construction, source registry, FileSource,
 *        and lifecycle event dispatch.
 *
 * @details
 * These tests do NOT load any WASM plugin.  They exercise the built-in
 * file: source and the lifecycle emit_* methods (which are no-ops when no
 * user plugins are loaded).
 */

#include <ttm/plugins/plugin_manager.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// =============================================================================
// Construction / destruction
// =============================================================================

TEST_CASE("PluginManager constructs and destructs without error", "[plugin_manager]") {
	CHECK_NOTHROW([]() { auto mgr = ttm::plugins::PluginManager::create().value(); }());
}

TEST_CASE("Multiple sequential PluginManager instances are safe", "[plugin_manager]") {
	{
		auto mgr1 = ttm::plugins::PluginManager::create().value();
		CHECK(mgr1.find_source("file:") != nullptr);
	}
	{
		auto mgr2 = ttm::plugins::PluginManager::create().value();
		CHECK(mgr2.find_source("file:") != nullptr);
	}
}

// =============================================================================
// Source registry
// =============================================================================

TEST_CASE("Built-in file: source is always registered", "[plugin_manager][source]") {
	auto mgr = ttm::plugins::PluginManager::create().value();
	CHECK(mgr.find_source("file:") != nullptr);
}

TEST_CASE("find_source returns nullptr for unknown scheme", "[plugin_manager][source]") {
	auto mgr = ttm::plugins::PluginManager::create().value();
	CHECK(mgr.find_source("hf:") == nullptr);
	CHECK(mgr.find_source("gh:") == nullptr);
	CHECK(mgr.find_source("") == nullptr);
	CHECK(mgr.find_source("s3:") == nullptr);
}

TEST_CASE("find_source is case-sensitive", "[plugin_manager][source]") {
	auto mgr = ttm::plugins::PluginManager::create().value();
	// "file:" is registered; "FILE:" and "File:" are not
	CHECK(mgr.find_source("FILE:") == nullptr);
	CHECK(mgr.find_source("File:") == nullptr);
}

// =============================================================================
// FileSource — open and read
// =============================================================================

namespace {

	/// RAII helper that creates a temporary file and removes it on destruction.
	struct TempFile {
		std::filesystem::path path;

		explicit TempFile(std::string_view content) {
			path = std::filesystem::temp_directory_path() / "ttm_test_XXXXXX.bin";
			// Use a fixed name derived from current time to avoid collisions.
			path.replace_filename(
					"ttm_plugin_test_" + std::to_string(std::hash<std::string>{}(std::string(content))) + ".bin"
			);
			std::ofstream f(path, std::ios::binary);
			f.write(content.data(), static_cast<std::streamsize>(content.size()));
		}

		~TempFile() { std::filesystem::remove(path); }

		/// URI in file:// form.
		std::string uri() const { return "file://" + path.string(); }
	};

} // anonymous namespace

TEST_CASE("FileSource opens and reads a local file", "[plugin_manager][source][file]") {
	constexpr std::string_view content = "hello from file source";
	TempFile tmp{content};

	auto mgr = ttm::plugins::PluginManager::create().value();
	auto* src = mgr.find_source("file:");
	REQUIRE(src != nullptr);

	auto reader = src->open(tmp.uri());
	REQUIRE(reader != nullptr);

	std::array<std::byte, 64> buf{};
	const auto n = reader->read(buf.data(), static_cast<std::streamsize>(buf.size()));
	REQUIRE(n == static_cast<std::streamsize>(content.size()));
	CHECK(std::string(reinterpret_cast<const char*>(buf.data()), static_cast<std::size_t>(n)) == content);
}

TEST_CASE("FileSource reaches EOF correctly", "[plugin_manager][source][file]") {
	TempFile tmp{"eof"};

	auto mgr = ttm::plugins::PluginManager::create().value();
	auto* src = mgr.find_source("file:");
	auto reader = src->open(tmp.uri());
	REQUIRE(reader != nullptr);

	std::array<std::byte, 128> buf{};
	reader->read(buf.data(), static_cast<std::streamsize>(buf.size())); // consume all
	CHECK(reader->read(buf.data(), static_cast<std::streamsize>(buf.size())) == 0);
}

TEST_CASE("FileSource returns nullptr for a missing file", "[plugin_manager][source][file]") {
	auto mgr = ttm::plugins::PluginManager::create().value();
	auto* src = mgr.find_source("file:");
	REQUIRE(src != nullptr);

	auto reader = src->open("file:///this/path/does/not/exist/at/all.bin");
	CHECK(reader == nullptr);
}

TEST_CASE("FileSource reader is seekable", "[plugin_manager][source][file]") {
	constexpr std::string_view content = "0123456789";
	TempFile tmp{content};

	auto mgr = ttm::plugins::PluginManager::create().value();
	auto reader = mgr.find_source("file:")->open(tmp.uri());
	REQUIRE(reader != nullptr);
	CHECK(reader->seekable());

	// Seek to position 5 and read 3 bytes
	reader->seek(5, std::ios_base::beg);
	std::array<std::byte, 4> buf{};
	const auto n = reader->read(buf.data(), 3);
	REQUIRE(n == 3);
	CHECK(std::string(reinterpret_cast<const char*>(buf.data()), 3) == "567");
}

TEST_CASE("FileSource reader exposes contents via as_stream()", "[plugin_manager][source][file]") {
	constexpr std::string_view content = "stream test content";
	TempFile tmp{content};

	auto mgr = ttm::plugins::PluginManager::create().value();
	auto reader = mgr.find_source("file:")->open(tmp.uri());
	REQUIRE(reader != nullptr);

	auto& stream = reader->as_stream();
	const std::string result{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>{}};
	CHECK(result == content);
}

// =============================================================================
// Lifecycle event dispatch (no-op with zero user plugins)
// =============================================================================

TEST_CASE("Lifecycle emit_* methods are no-ops when no user plugins are loaded", "[plugin_manager][lifecycle]") {
	auto mgr = ttm::plugins::PluginManager::create().value();

	CHECK_NOTHROW(mgr.emit_fit_begin("{}"));
	CHECK_NOTHROW(mgr.emit_epoch_begin(0, 10));
	CHECK_NOTHROW(mgr.emit_batch_begin(0, 100));
	CHECK(mgr.emit_loss_computed(1.5f) == Catch::Approx(1.5f));
	CHECK_NOTHROW(mgr.emit_batch_end(0, 1.5f, "{}"));
	CHECK_FALSE(mgr.emit_epoch_end(0, "{}")); // no early stop requested
	CHECK_NOTHROW(mgr.emit_validation_end("{}"));
	CHECK_NOTHROW(mgr.emit_fit_end("{}"));
}

TEST_CASE("emit_loss_computed returns input unchanged with no user plugins", "[plugin_manager][lifecycle]") {
	auto mgr = ttm::plugins::PluginManager::create().value();
	CHECK(mgr.emit_loss_computed(0.0f) == Catch::Approx(0.0f));
	CHECK(mgr.emit_loss_computed(3.14f) == Catch::Approx(3.14f));
	CHECK(mgr.emit_loss_computed(-1.0f) == Catch::Approx(-1.0f));
}

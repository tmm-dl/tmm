/**
 * @file test_byte_reader.cpp
 * @brief Unit tests for IByteReader and the ByteReaderBuf std::istream adapter.
 *
 * @details
 * Uses a StringReader (a trivial in-memory IByteReader) to exercise ByteReaderBuf
 * without requiring any real plugin or filesystem access.
 */

#include <ttm/plugins/extension.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <ios>
#include <iosfwd>
#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace {

	/**
	 * @brief Concrete IByteReader backed by a std::string for testing.
	 */
	class StringReader final : public ttm::plugins::IByteReader {
	public:
		explicit StringReader(std::string data) : data_(std::move(data)) {}

		std::streamsize read(std::byte* buf, std::streamsize n) override {
			const auto remaining = static_cast<std::streamsize>(data_.size()) - pos_;
			if (remaining <= 0) {
				return 0;
			}
			const auto to_read = std::min(n, remaining);
			std::memcpy(buf, data_.data() + static_cast<std::size_t>(pos_), static_cast<std::size_t>(to_read));
			pos_ += to_read;
			return to_read;
		}

		[[nodiscard]] bool seekable() const noexcept override { return true; }

		std::streampos seek(std::streamoff off, std::ios_base::seekdir dir) override {
			std::streamsize new_pos{};
			switch (dir) {
			case std::ios_base::beg:
				new_pos = off;
				break;
			case std::ios_base::cur:
				new_pos = pos_ + off;
				break;
			case std::ios_base::end:
				new_pos = static_cast<std::streamsize>(data_.size()) + off;
				break;
			default:
				return {-1};
			}
			if (new_pos < 0 || new_pos > static_cast<std::streamsize>(data_.size())) {
				return {-1};
			}
			pos_ = new_pos;
			return {pos_};
		}

	private:
		std::string data_;
		std::streamsize pos_ = 0;
	};

	/**
	 * @brief Concrete IByteReader that is never seekable, for negative-path tests.
	 */
	class UnseekableReader final : public ttm::plugins::IByteReader {
	public:
		explicit UnseekableReader(std::string data) : data_(std::move(data)) {}

		std::streamsize read(std::byte* buf, std::streamsize n) override {
			const auto remaining = static_cast<std::streamsize>(data_.size()) - pos_;
			if (remaining <= 0) {
				return 0;
			}
			const auto to_read = std::min(n, remaining);
			std::memcpy(buf, data_.data() + static_cast<std::size_t>(pos_), static_cast<std::size_t>(to_read));
			pos_ += to_read;
			return to_read;
		}

		[[nodiscard]] bool seekable() const noexcept override { return false; }

		std::streampos seek(std::streamoff /*off*/, std::ios_base::seekdir /*dir*/) override { return {-1}; }

	private:
		std::string data_;
		std::streamsize pos_ = 0;
	};

} // anonymous namespace

// =============================================================================
// Direct read / seek tests
// =============================================================================

TEST_CASE("StringReader reads all bytes in one call", "[byte_reader]") {
	StringReader reader("hello");
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers) -- arbitrary buffer that is big
	// enough for the test case
	std::array<std::byte, 16> buf{};
	const auto numread = reader.read(buf.data(), static_cast<std::streamsize>(buf.size()));
	REQUIRE(numread == 5);
	CHECK(std::memcmp(buf.data(), "hello", 5) == 0);
}

TEST_CASE("StringReader returns 0 at EOF", "[byte_reader]") {
	StringReader reader("x");
	std::array<std::byte, 4> buf{};
	reader.read(buf.data(), 4); // consume
	CHECK(reader.read(buf.data(), 4) == 0);
}

TEST_CASE("StringReader reads partial data when buf is smaller than content", "[byte_reader]") {
	StringReader reader("abcdefgh");
	std::array<std::byte, 3> buf{};
	CHECK(reader.read(buf.data(), 3) == 3);
	CHECK(std::memcmp(buf.data(), "abc", 3) == 0);
	CHECK(reader.read(buf.data(), 3) == 3);
	CHECK(std::memcmp(buf.data(), "def", 3) == 0);
}

TEST_CASE("StringReader seek SEEK_SET repositions correctly", "[byte_reader]") {
	StringReader reader("0123456789");
	std::array<std::byte, 4> buf{};

	const auto pos = reader.seek(5, std::ios_base::beg);
	REQUIRE(pos == std::streampos(5));
	reader.read(buf.data(), 3);
	CHECK(std::memcmp(buf.data(), "567", 3) == 0);
}

TEST_CASE("StringReader seek SEEK_CUR advances from current position", "[byte_reader]") {
	StringReader reader("0123456789");
	std::array<std::byte, 2> buf{};

	reader.read(buf.data(), 2); // advance to pos 2
	const auto pos = reader.seek(3, std::ios_base::cur);
	REQUIRE(pos == std::streampos(5));
	reader.read(buf.data(), 2);
	CHECK(std::memcmp(buf.data(), "56", 2) == 0);
}

TEST_CASE("StringReader seek SEEK_END positions from end", "[byte_reader]") {
	StringReader reader("0123456789");
	std::array<std::byte, 2> buf{};

	const auto pos = reader.seek(-3, std::ios_base::end);
	REQUIRE(pos == std::streampos(7));
	reader.read(buf.data(), 2);
	CHECK(std::memcmp(buf.data(), "78", 2) == 0);
}

TEST_CASE("StringReader seek out of bounds returns -1", "[byte_reader]") {
	StringReader reader("abc");
	CHECK(reader.seek(-1, std::ios_base::beg) == std::streampos(-1));
	CHECK(reader.seek(100, std::ios_base::beg) == std::streampos(-1));
}

TEST_CASE("UnseekableReader::seekable returns false", "[byte_reader]") {
	UnseekableReader reader("data");
	CHECK_FALSE(reader.seekable());
	CHECK(reader.seek(0, std::ios_base::beg) == std::streampos(-1));
}

// =============================================================================
// as_stream() / ByteReaderBuf tests
// =============================================================================

TEST_CASE("as_stream reads words via std::istream operator>>", "[byte_reader][as_stream]") {
	StringReader reader("hello world");
	auto& stream = reader.as_stream();
	std::string word1;
	std::string word2;
	stream >> word1 >> word2;
	CHECK(word1 == "hello");
	CHECK(word2 == "world");
}

TEST_CASE("as_stream returns the same object on repeated calls", "[byte_reader][as_stream]") {
	StringReader reader("data");
	auto& str1 = reader.as_stream();
	auto& str2 = reader.as_stream();
	CHECK(&str1 == &str2);
}

TEST_CASE("as_stream reads multi-line text correctly", "[byte_reader][as_stream]") {
	StringReader reader("line1\nline2\nline3");
	auto& stream = reader.as_stream();

	std::string line;
	REQUIRE(std::getline(stream, line));
	CHECK(line == "line1");
	REQUIRE(std::getline(stream, line));
	CHECK(line == "line2");
	REQUIRE(std::getline(stream, line));
	CHECK(line == "line3");
	CHECK_FALSE(std::getline(stream, line)); // EOF
}

TEST_CASE("as_stream reports EOF correctly", "[byte_reader][as_stream]") {
	StringReader reader("x");
	auto& stream = reader.as_stream();
	char c = 0;
	stream >> c;
	CHECK(c == 'x');
	stream >> c;
	CHECK(stream.eof());
}

TEST_CASE("as_stream supports seekg when reader is seekable", "[byte_reader][as_stream]") {
	StringReader reader("abcde");
	auto& stream = reader.as_stream();

	stream.seekg(2, std::ios_base::beg);
	char c = 0;
	stream >> c;
	CHECK(c == 'c');
}

TEST_CASE("as_stream seekg fails gracefully on non-seekable reader", "[byte_reader][as_stream]") {
	UnseekableReader reader("abcde");
	auto& stream = reader.as_stream();

	stream.seekg(2, std::ios_base::beg);
	// Stream should be in a failed state after an unsupported seek
	CHECK(stream.fail());
}

/**
 * @file logger_tensorboard.cpp
 * @brief TensorBoard callback — writes TFEvents binary files.
 *
 * @details
 * Writes one `events.out.tfevents.*` file per run under the configured log
 * directory.  Each call to on_epoch_end emits one scalar Summary event per
 * metric in the metrics JSON object.
 *
 * The TFEvents binary format:
 *   Record = [uint64 data_len][uint32 masked_crc32c(data_len)]
 *            [data_len bytes][uint32 masked_crc32c(data)]
 *
 * The data payload is a serialised protobuf Event message (hand-rolled
 * encoding — no protobuf library dependency).
 *
 * CRC variant: Castagnoli (CRC32C), polynomial 0x82F63B78.
 */

#include "loggers.hpp"

#include <ttm/plugins/abi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace {

/* ============================================================================
 * Minimal JSON helpers
 * ========================================================================== */

static std::string jsonGetStr(std::string_view json, std::string_view key,
                               std::string_view def = {}) {
	const std::string needle = std::string("\"") + std::string(key) + "\":";
	const auto pos = json.find(needle);
	if (pos == std::string_view::npos) return std::string(def);
	auto val = json.substr(pos + needle.size());
	while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.remove_prefix(1);
	if (val.empty()) return std::string(def);
	if (val.front() == '"') {
		val.remove_prefix(1);
		const auto end = val.find('"');
		return end == std::string_view::npos ? std::string(def) : std::string(val.substr(0, end));
	}
	const auto end = val.find_first_of(",}] \t\n");
	return std::string(val.substr(0, end == std::string_view::npos ? val.size() : end));
}

/// Iterate all (key, float) pairs in a flat JSON object, calling fn for each.
static void jsonForEachMetric(std::string_view json,
                               void (*fn)(std::string_view key, float val, void* ud),
                               void* ud) {
	auto pos = json.find('{');
	if (pos == std::string_view::npos) return;
	json.remove_prefix(pos + 1);

	while (!json.empty()) {
		while (!json.empty() && (json.front() == ' ' || json.front() == '\t'
		                         || json.front() == '\n' || json.front() == ','))
			json.remove_prefix(1);
		if (json.empty() || json.front() == '}') break;
		if (json.front() != '"') break;

		json.remove_prefix(1);
		const auto keyEnd = json.find('"');
		if (keyEnd == std::string_view::npos) break;
		const auto key = json.substr(0, keyEnd);
		json.remove_prefix(keyEnd + 1);

		while (!json.empty() && (json.front() == ' ' || json.front() == ':')) json.remove_prefix(1);

		const auto valEnd = json.find_first_of(",}");
		if (valEnd == std::string_view::npos) break;
		const auto valStr = std::string(json.substr(0, valEnd));
		json.remove_prefix(valEnd);
		try {
			const float val = std::stof(valStr);
			fn(key, val, ud);
		} catch (...) {}
	}
}

/* ============================================================================
 * CRC32C (Castagnoli) — polynomial 0x82F63B78
 * ========================================================================== */

static std::array<uint32_t, 256> buildCrc32cTable() {
	std::array<uint32_t, 256> t{};
	for (uint32_t i = 0; i < 256; ++i) {
		uint32_t crc = i;
		for (int j = 0; j < 8; ++j)
			crc = (crc >> 1) ^ ((crc & 1u) ? 0x82F63B78u : 0u);
		t[i] = crc;
	}
	return t;
}

static const std::array<uint32_t, 256> kCrc32cTable = buildCrc32cTable();

static uint32_t crc32c(const uint8_t* data, size_t len) {
	uint32_t crc = 0xFFFFFFFFu;
	for (size_t i = 0; i < len; ++i)
		crc = (crc >> 8) ^ kCrc32cTable[(crc ^ data[i]) & 0xFFu];
	return crc ^ 0xFFFFFFFFu;
}

static uint32_t maskedCrc32c(const uint8_t* data, size_t len) {
	const uint32_t c = crc32c(data, len);
	return ((c >> 15u) | (c << 17u)) + 0xa282ead8ul;
}

/* ============================================================================
 * Minimal protobuf encoding helpers
 * ========================================================================== */

static void pbAppendVarint(std::vector<uint8_t>& buf, uint64_t v) {
	while (v >= 0x80u) {
		buf.push_back(static_cast<uint8_t>(v | 0x80u));
		v >>= 7u;
	}
	buf.push_back(static_cast<uint8_t>(v));
}

/// Wire type 1 (64-bit fixed): encodes a double or uint64 as 8 LE bytes.
static void pbFixed64Field(std::vector<uint8_t>& buf, int fieldNum, uint64_t v) {
	pbAppendVarint(buf, (static_cast<uint64_t>(fieldNum) << 3u) | 1u);
	for (int i = 0; i < 8; ++i) { buf.push_back(static_cast<uint8_t>(v & 0xFFu)); v >>= 8u; }
}

/// Wire type 0 (varint): encodes a non-negative int64.
static void pbVarintField(std::vector<uint8_t>& buf, int fieldNum, int64_t v) {
	pbAppendVarint(buf, (static_cast<uint64_t>(fieldNum) << 3u) | 0u);
	pbAppendVarint(buf, static_cast<uint64_t>(v));
}

/// Wire type 2 (length-delimited): embeds a sub-message or bytes.
static void pbBytesField(std::vector<uint8_t>& buf, int fieldNum,
                          const std::vector<uint8_t>& data) {
	pbAppendVarint(buf, (static_cast<uint64_t>(fieldNum) << 3u) | 2u);
	pbAppendVarint(buf, static_cast<uint64_t>(data.size()));
	buf.insert(buf.end(), data.begin(), data.end());
}

static void pbStringField(std::vector<uint8_t>& buf, int fieldNum, std::string_view s) {
	pbAppendVarint(buf, (static_cast<uint64_t>(fieldNum) << 3u) | 2u);
	pbAppendVarint(buf, static_cast<uint64_t>(s.size()));
	buf.insert(buf.end(), s.begin(), s.end());
}

/// Wire type 5 (32-bit fixed): encodes a float as 4 LE bytes.
static void pbFloatField(std::vector<uint8_t>& buf, int fieldNum, float v) {
	pbAppendVarint(buf, (static_cast<uint64_t>(fieldNum) << 3u) | 5u);
	uint32_t bits = 0;
	std::memcpy(&bits, &v, 4);
	for (int i = 0; i < 4; ++i) { buf.push_back(static_cast<uint8_t>(bits & 0xFFu)); bits >>= 8u; }
}

/* ============================================================================
 * TFEvents record building
 *
 * Protobuf schema (field numbers from tensorflow/core/util/events_writer.h):
 *   Event    { wall_time:1(double), step:2(int64), summary:5(Summary),
 *              file_version:7(string) }
 *   Summary  { value:1(repeated Value) }
 *   Value    { tag:1(string), simple_value:2(float) }
 * ========================================================================== */

static bool tfWriteRecord(FILE* f, const std::vector<uint8_t>& data) {
	const uint64_t len = static_cast<uint64_t>(data.size());
	const uint32_t lenCrc  = maskedCrc32c(reinterpret_cast<const uint8_t*>(&len), 8);
	const uint32_t dataCrc = maskedCrc32c(data.data(), data.size());
	if (std::fwrite(&len,    8, 1, f) != 1) return false;
	if (std::fwrite(&lenCrc, 4, 1, f) != 1) return false;
	if (std::fwrite(data.data(), 1, data.size(), f) != data.size()) return false;
	if (std::fwrite(&dataCrc, 4, 1, f) != 1) return false;
	return true;
}

static std::vector<uint8_t> tfVersionEvent(double wallTime) {
	std::vector<uint8_t> ev;
	uint64_t wt = 0; std::memcpy(&wt, &wallTime, 8);
	pbFixed64Field(ev, 1, wt);
	pbStringField(ev, 7, "brain.Event:2");
	return ev;
}

static std::vector<uint8_t> tfScalarEvent(double wallTime, int64_t step,
                                           std::string_view tag, float value) {
	std::vector<uint8_t> summaryValue;
	pbStringField(summaryValue, 1, tag);
	pbFloatField(summaryValue,  2, value);

	std::vector<uint8_t> summary;
	pbBytesField(summary, 1, summaryValue);

	std::vector<uint8_t> ev;
	uint64_t wt = 0; std::memcpy(&wt, &wallTime, 8);
	pbFixed64Field(ev, 1, wt);
	pbVarintField(ev,  2, step);
	pbBytesField(ev,   5, summary);
	return ev;
}

/* ============================================================================
 * Slot allocator
 * ========================================================================== */

constexpr int kMaxTbSlots = 64;

struct TensorBoardState {
	char    logDir[512] = "runs";
	FILE*   eventsFile  = nullptr;
	int64_t globalStep  = 0;
};

static TensorBoardState s_tb_slots[kMaxTbSlots];
static bool             s_tb_used[kMaxTbSlots] = {};

static ttm_handle tbAlloc() {
	for (int i = 0; i < kMaxTbSlots; ++i) {
		if (!s_tb_used[i]) {
			s_tb_used[i] = true;
			s_tb_slots[i] = TensorBoardState{};
			return static_cast<ttm_handle>(i);
		}
	}
	return TTM_INVALID_HANDLE;
}

static TensorBoardState* tbGet(ttm_handle h) {
	if (h < 0 || h >= kMaxTbSlots || !s_tb_used[static_cast<int>(h)]) return nullptr;
	return &s_tb_slots[static_cast<int>(h)];
}

/* ============================================================================
 * Callback implementation
 * ========================================================================== */

static ttm_handle tensorBoardCreate(const char* cfg, uint32_t cfgLen) {
	const auto h = tbAlloc();
	if (h == TTM_INVALID_HANDLE) return TTM_INVALID_HANDLE;
	auto* st = tbGet(h);
	const auto dir = jsonGetStr(std::string_view{cfg, cfgLen}, "logDir", "runs");
	std::strncpy(st->logDir, dir.c_str(), sizeof(st->logDir) - 1);
	return h;
}

static void tensorBoardOnFitBegin(ttm_handle h, const char* /*metrics*/, uint32_t /*len*/) {
	auto* st = tbGet(h);
	if (st == nullptr) return;
	st->globalStep = 0;

	std::error_code ec;
	std::filesystem::create_directories(st->logDir, ec);

	const auto ts = static_cast<long long>(std::time(nullptr));
	const std::string fname = std::string(st->logDir)
	                        + "/events.out.tfevents."
	                        + std::to_string(ts);
	st->eventsFile = std::fopen(fname.c_str(), "wb");
	if (st->eventsFile == nullptr) return;

	const auto rec = tfVersionEvent(static_cast<double>(ts));
	tfWriteRecord(st->eventsFile, rec);
	std::fflush(st->eventsFile);
}

struct TbScalarCtx { FILE* f; double wallTime; int64_t step; };

static void tbWriteScalar(std::string_view key, float val, void* ud) {
	auto* ctx = static_cast<TbScalarCtx*>(ud);
	const auto rec = tfScalarEvent(ctx->wallTime, ctx->step, key, val);
	tfWriteRecord(ctx->f, rec);
}

static int32_t tensorBoardOnEpochEnd(ttm_handle h, int64_t epoch,
                                      const char* metricsJson, uint32_t len) {
	auto* st = tbGet(h);
	if (st == nullptr || st->eventsFile == nullptr) return 0;
	st->globalStep = epoch + 1;
	const double wallTime = static_cast<double>(std::time(nullptr));
	TbScalarCtx ctx{st->eventsFile, wallTime, st->globalStep};
	jsonForEachMetric({metricsJson, len}, tbWriteScalar, &ctx);
	std::fflush(st->eventsFile);
	return 0;
}

static void tensorBoardOnFitEnd(ttm_handle h, const char* metricsJson, uint32_t len) {
	auto* st = tbGet(h);
	if (st == nullptr || st->eventsFile == nullptr) return;
	// Write final metrics at the last step
	const double wallTime = static_cast<double>(std::time(nullptr));
	TbScalarCtx ctx{st->eventsFile, wallTime, st->globalStep};
	jsonForEachMetric({metricsJson, len}, tbWriteScalar, &ctx);
	std::fclose(st->eventsFile);
	st->eventsFile = nullptr;
}

static void tensorBoardDestroy(ttm_handle h) {
	auto* st = tbGet(h);
	if (st != nullptr && st->eventsFile != nullptr) {
		std::fclose(st->eventsFile);
		st->eventsFile = nullptr;
	}
	if (h >= 0 && h < kMaxTbSlots) s_tb_used[static_cast<int>(h)] = false;
}

} // anonymous namespace

ttm_trainer_callback_vtable g_cb_tensorBoard = {
	/* create         */ tensorBoardCreate,
	/* on_fit_begin   */ tensorBoardOnFitBegin,
	/* on_epoch_begin */ nullptr,
	/* on_epoch_end   */ tensorBoardOnEpochEnd,
	/* on_fit_end     */ tensorBoardOnFitEnd,
	/* destroy        */ tensorBoardDestroy,
};

void tbTeardownSlots() {
	for (int i = 0; i < kMaxTbSlots; ++i) {
		if (s_tb_used[i] && s_tb_slots[i].eventsFile != nullptr) {
			std::fclose(s_tb_slots[i].eventsFile);
			s_tb_slots[i].eventsFile = nullptr;
		}
		s_tb_used[i] = false;
	}
}

/**
 * @file console_ui.cpp
 * @brief TTM console-UI plugin — WASM/WASI, FTXUI DOM-only rendering.
 *
 * @details
 * Displays a live full-screen TUI during training with three panels:
 *
 *  ┌ TTM Training ─────────────────────────────┐
 *  │ Model: …  Dataset: …  Epoch: 3/10  Step: … │
 *  └────────────────────────────────────────────┘
 *  ┌ Metrics ────────────────────────────────────┐
 *  │ train_loss  ▂▄▆█▇▅▃   0.4231               │
 *  │ val_loss    ▂▄▅▆▅▄▃   0.5128               │
 *  └────────────────────────────────────────────┘
 *  ┌ Logs ───────────────────────────────────────┐
 *  │ [INFO]  epoch 3/10 — loss: 0.4231  …        │
 *  └────────────────────────────────────────────┘
 *
 * Uses FTXUI ftxui_screen + ftxui_dom (no ScreenInteractive — no pthreads).
 * Terminal size is obtained from the host via the ttm_terminal_size import.
 * FTXUI is compiled with -D__EMSCRIPTEN__=1 so Terminal::Size() returns the
 * safe default {80,24} rather than calling ioctl; we always pass an explicit
 * Dimension::Fixed(w,h) to Screen::Create() so the actual terminal dimensions
 * are used at every redraw.
 *
 * Rendering strategy: on each significant event, emit ANSI cursor-home
 * (\033[H) and overwrite the previous frame in-place.  This avoids flicker
 * from a full clear while still keeping the display current.
 */

#include <ttm/plugins/abi.h>

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/screen/screen.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

/* =========================================================================
 * Host imports
 *
 * These are imported from the "ttm" WAMR module namespace.  Signatures must
 * match the NativeSymbol table registered in wasm_loader.cpp.
 * ====================================================================== */

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) -- WASM import attribute requires external linkage
__attribute__((import_module("ttm"), import_name("ttm_terminal_size")))
extern void ttm_terminal_size(uint32_t* out_width, uint32_t* out_height);

/* =========================================================================
 * Internal state
 * ====================================================================== */

namespace {

	constexpr int kMaxLogs    = 200;
	constexpr int kMaxHistory = 80;

	struct LogEntry {
		uint32_t    level;
		std::string message;
	};

	struct MetricHistory {
		std::deque<float> values;
		float             vmin = 0.0f;
		float             vmax = 1.0f;

		void push(float v) {
			if (static_cast<int>(values.size()) >= kMaxHistory) values.pop_front();
			values.push_back(v);
			vmin = vmax = values.front();
			for (float x : values) {
				vmin = std::min(vmin, x);
				vmax = std::max(vmax, x);
			}
		}
	};

	std::deque<LogEntry>             g_logs;
	std::map<std::string, MetricHistory> g_metrics; // std::map avoids hash-table __next_prime overflow in 32-bit WASM
	std::vector<std::string>                       g_metric_keys; ///< insertion-ordered keys

	std::string g_model_path;
	std::string g_dataset_uri;
	std::string g_dataset_split;
	uint32_t    g_current_epoch = 0;
	uint32_t    g_total_epochs  = 0;
	int32_t     g_global_step   = 0;
	bool        g_fit_started   = false;

	/* -----------------------------------------------------------------------
	 * Utilities
	 * -------------------------------------------------------------------- */

	/** @brief Extract a JSON string value: "key":"value". */
	std::string json_str(const std::string& json, const std::string& key) {
		const std::string search = "\"" + key + "\":\"";
		const auto        pos    = json.find(search);
		if (pos == std::string::npos) return {};
		const auto start = pos + search.size();
		const auto end   = json.find('"', start);
		if (end == std::string::npos) return {};
		return json.substr(start, end - start);
	}

	/** @brief Extract a JSON integer value: "key":digits. */
	int64_t json_int(const std::string& json, const std::string& key) {
		const std::string search = "\"" + key + "\":";
		const auto        pos    = json.find(search);
		if (pos == std::string::npos) return 0;
		auto p = pos + search.size();
		bool neg = false;
		if (p < json.size() && json[p] == '-') { neg = true; ++p; }
		int64_t v = 0;
		for (; p < json.size() && json[p] >= '0' && json[p] <= '9'; ++p)
			v = v * 10 + static_cast<int64_t>(json[p] - '0');
		return neg ? -v : v;
	}

	/** @brief Strip ANSI escape sequences from a string view. */
	std::string strip_ansi(const std::string& s) {
		std::string out;
		out.reserve(s.size());
		bool in_esc = false;
		for (char c : s) {
			if (c == '\033') { in_esc = true; continue; }
			if (in_esc) {
				if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) in_esc = false;
				continue;
			}
			if (c == '\n' || c == '\r') continue; // flatten multi-line to single
			out += c;
		}
		return out;
	}

	/* -----------------------------------------------------------------------
	 * DOM builders
	 * -------------------------------------------------------------------- */

	using namespace ftxui;

	Element make_header() {
		const std::string epoch_str = g_fit_started
		                                  ? (std::to_string(g_current_epoch) + "/" +
		                                     std::to_string(g_total_epochs))
		                                  : "—";

		return window(
				text(" TTM Training ") | bold,
				vbox({
						hbox({
								text("Model:   ") | bold,
								text(g_model_path.empty() ? "—" : g_model_path) | color(Color::Cyan) | flex,
								separator(),
								text(" Epoch: ") | bold,
								text(epoch_str) | color(Color::Yellow),
						}),
						hbox({
								text("Dataset: ") | bold,
								text(g_dataset_uri.empty() ? "—" : g_dataset_uri) | color(Color::Cyan) | flex,
								separator(),
								text(" Split: ") | bold,
								text(g_dataset_split.empty() ? "—" : g_dataset_split),
						}),
						hbox({
								text("Step:    ") | bold,
								text(std::to_string(g_global_step)) | color(Color::Green),
								filler(),
						}),
				})
		);
	}

	Element make_metrics_panel() {
		Elements rows;

		for (const auto& key : g_metric_keys) {
			const auto it = g_metrics.find(key);
			if (it == g_metrics.end()) continue;
			const MetricHistory& h = it->second;
			if (h.values.empty()) continue;

			const float latest = h.values.back();
			const float vmin   = h.vmin;
			const float vmax   = h.vmax;
			const int   n      = static_cast<int>(h.values.size());

			/* Sparkline via ftxui::graph */
			auto sparkline = graph([vals = std::vector<float>(h.values.begin(), h.values.end()),
			                        vmin, vmax](int w, int ht) {
				std::vector<int> out(static_cast<size_t>(w), 0);
				const int        count  = static_cast<int>(vals.size());
				const float      range  = vmax - vmin;
				for (int i = 0; i < w; ++i) {
					const int src = count - w + i;
					if (src < 0 || src >= count) continue;
					float norm = (range > 1e-9f) ? (vals[static_cast<size_t>(src)] - vmin) / range : 0.5f;
					norm = std::clamp(norm, 0.0f, 1.0f);
					out[static_cast<size_t>(i)] = static_cast<int>(norm * static_cast<float>(ht));
				}
				return out;
			}) | color(Color::GreenLight)
			   | flex;

			/* Format latest value */
			char buf[24]{};
			if (latest >= 1e3f || (latest != 0.0f && latest < 1e-3f))
				snprintf(buf, sizeof(buf), "%.3e", static_cast<double>(latest));
			else
				snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(latest));

			rows.push_back(
					hbox({
							text(key) | color(Color::Cyan) | size(WIDTH, EQUAL, 16),
							text(" "),
							sparkline,
							text(" "),
							text(buf) | color(Color::Yellow) | size(WIDTH, EQUAL, 10),
					}) | size(HEIGHT, EQUAL, 3)
			);
		}

		if (rows.empty()) {
			rows.push_back(text("  No metrics yet — waiting for training to start…") | dim);
		}

		return window(text(" Metrics ") | bold, vbox(std::move(rows))) | flex;
	}

	Element make_logs_panel() {
		Elements lines;

		const int total    = static_cast<int>(g_logs.size());
		const int max_show = std::max(1, total);
		const int start    = std::max(0, total - max_show);

		for (int i = start; i < total; ++i) {
			const LogEntry& e = g_logs[static_cast<size_t>(i)];

			Element prefix;
			switch (e.level) {
			case TTM_LOG_TRACE: prefix = text("[TRACE] ") | color(Color::GrayDark); break;
			case TTM_LOG_DEBUG: prefix = text("[DEBUG] ") | color(Color::Blue); break;
			case TTM_LOG_INFO: prefix = text("[INFO]  ") | color(Color::Green); break;
			case TTM_LOG_WARN: prefix = text("[WARN]  ") | color(Color::Yellow); break;
			case TTM_LOG_ERROR: prefix = text("[ERROR] ") | color(Color::Red) | bold; break;
			default: prefix = text("        "); break;
			}
			lines.push_back(hbox({prefix, text(strip_ansi(e.message))}));
		}

		if (lines.empty()) {
			lines.push_back(text("  No log messages yet…") | dim);
		}

		return window(text(" Logs ") | bold, vbox(std::move(lines)) | frame | flex) | flex;
	}

	void redraw() {
		uint32_t w = 80, h = 24;
		ttm_terminal_size(&w, &h);
		if (w < 20) w = 20;
		if (h < 8) h = 8;

		const int cols = static_cast<int>(w);
		const int rows = static_cast<int>(h);

		auto doc = vbox({
				              make_header(),
				              make_metrics_panel(),
				              make_logs_panel(),
			              }) |
		           size(HEIGHT, EQUAL, rows) | size(WIDTH, EQUAL, cols);

		Screen screen(cols, rows);
		Render(screen, doc);

		/* Cursor to top-left, then overwrite frame in-place (no full-clear flicker) */
		fputs("\033[H", stdout);
		fputs(screen.ToString().c_str(), stdout);
		fflush(stdout);
	}

} // anonymous namespace

/* =========================================================================
 * Required plugin exports
 * ====================================================================== */

extern "C" {

static ttm_plugin_info g_info = {
		TTM_ABI_VERSION, "console-ui", "0.1.0",
		"Live TUI training dashboard — FTXUI DOM, WASI"};

ttm_plugin_info* ttm_plugin_get_info(void) { return &g_info; }

ttm_error ttm_plugin_init(
		const ttm_host_api* /*host*/, const char* /*cfg*/, uint32_t /*len*/
) {
	/* Reserve terminal area: clear screen once, then use cursor-home overwrites. */
	fputs("\033[2J\033[H", stdout);
	fflush(stdout);
	redraw(); /* Show empty dashboard immediately. */
	return TTM_OK;
}

void ttm_plugin_teardown(void) {
	/* Position cursor below the TUI so the shell prompt appears cleanly. */
	uint32_t w = 80, h = 24;
	ttm_terminal_size(&w, &h);
	char buf[32]{};
	snprintf(buf, sizeof(buf), "\033[%u;0H\n", h);
	fputs(buf, stdout);
	fflush(stdout);
}

/* =========================================================================
 * Optional lifecycle hooks
 * ====================================================================== */

void ttm_on_fit_begin(const char* ctx_json, uint32_t len) {
	g_fit_started   = true;
	g_current_epoch = 0;
	g_global_step   = 0;

	const std::string json(ctx_json, len);
	g_dataset_uri   = json_str(json, "dataset_uri");
	g_dataset_split = json_str(json, "dataset_split");
	g_model_path    = json_str(json, "model_path");
	g_total_epochs  = static_cast<uint32_t>(json_int(json, "total_epochs"));

	redraw();
}

void ttm_on_epoch_begin(uint32_t epoch, uint32_t /*total*/) {
	g_current_epoch = epoch;
	redraw();
}

void ttm_on_batch_end(
		uint32_t /*batch*/, float /*loss*/, const char* /*json*/, uint32_t /*len*/
) {
	/* Throttle redraws to every 10 batches to reduce output volume. */
	static uint32_t s_count = 0;
	if ((++s_count % 10u) == 0u) redraw();
}

int32_t ttm_on_epoch_end(uint32_t epoch, const char* /*json*/, uint32_t /*len*/) {
	g_current_epoch = epoch;
	redraw();
	return 0; /* 0 = do not request early stopping */
}

void ttm_on_fit_end(const char* /*json*/, uint32_t /*len*/) { redraw(); }

/* =========================================================================
 * Log + metric receive hooks
 * ====================================================================== */

void ttm_on_log(uint32_t level, const char* msg, uint32_t len) {
	if (static_cast<int>(g_logs.size()) >= kMaxLogs) g_logs.pop_front();
	g_logs.push_back({level, std::string(msg, len)});
	redraw();
}

void ttm_on_metric(const char* key, uint32_t key_len, float value, int32_t step) {
	g_global_step = step;

	const std::string k(key, key_len);
	if (g_metrics.find(k) == g_metrics.end()) {
		g_metric_keys.push_back(k);
	}
	g_metrics[k].push(value);
	redraw();
}

} // extern "C"

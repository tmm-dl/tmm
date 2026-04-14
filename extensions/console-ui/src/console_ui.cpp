/**
 * @file console_ui.cpp
 * @brief TMM console-UI plugin — two rendering modes.
 *
 * ### DOM-only mode  (default / WASM build)
 * Renders a full-screen ANSI frame on every significant event by writing
 * directly to stdout.  No threads, no pthreads, WASI-compatible.
 *
 * ### Interactive mode  (native build, config `"interactive":true`)
 * Uses `ftxui::ScreenInteractive::Fullscreen()` running in a dedicated render
 * thread.  Callbacks post a custom event that wakes the event loop; all shared
 * state is protected by a mutex.
 *
 * Layout (interactive):
 * ```
 * ┌ Utilization ─────────────────────────────────────────────────────────┐
 * │ RAM: |████████████░░░░░░░░░░░░░░░░░░░|  1.2 GB / 62 GB              │
 * └──────────────────────────────────────────────────────────────────────┘
 * ┌ [ Metrics ] [ Logs ] ───────────────┐ ┌ Info ─────────────────────── ┐
 * │  train_loss  ▂▄▆█▇▅▃  0.4231       │ │ Model:   model.py             │
 * │  val_loss    ▂▄▅▆▅▄▃  0.5128       │ │ Dataset: hf:thagen/SCITE      │
 * │                                     │ │ Split:   train                │
 * │  (or log lines)                     │ │ Epochs:  3 / 5               │
 * │                                     │ │ Step:    1 024                │
 * └─────────────────────────────────────┘ └──────────────────────────────┘
 * ┌ Progress ────────────────────────────────────────────────────────────┐
 * │ Epoch 3/5  |███████████████░░░░░░░░░░░░░░░░░|  45.2 %               │
 * └──────────────────────────────────────────────────────────────────────┘
 * ```
 */

#include <tmm/plugins/abi.h>

/* FTXUI DOM/Screen — available in both native and WASM builds. */
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include <ftxui/screen/screen.hpp>

/* FTXUI Component API — native only (WASI has no pthreads / signals). */
#ifndef __EMSCRIPTEN__
#include <atomic>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <mutex>
#include <thread>
#ifdef __linux__
#include <sys/resource.h>
#include <sys/sysinfo.h>
#endif
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

/* =========================================================================
 * Shared state (all fields guarded by g_mtx in interactive mode)
 * ====================================================================== */

namespace {

	constexpr int kMaxLogs = 200;
	constexpr int kMaxHistory = 80;

	struct LogEntry {
		uint32_t level;
		std::string message;
	};

	struct MetricHistory {
		std::deque<float> values;
		float vmin = 0.0f;
		float vmax = 1.0f;

		void push(float v) {
			if (static_cast<int>(values.size()) >= kMaxHistory)
				values.pop_front();
			values.push_back(v);
			vmin = vmax = values.front();
			for (float x : values) {
				vmin = std::min(vmin, x);
				vmax = std::max(vmax, x);
			}
		}
	};

	/* All mutable globals — zero-initialised. */
	std::deque<LogEntry> g_logs;
	std::map<std::string, MetricHistory> g_metrics;
	std::vector<std::string> g_metricKeys; ///< insertion-ordered
	std::string g_modelPath;
	std::string g_datasetUri;
	std::string g_datasetSplit;
	uint32_t g_currentEpoch = 0;
	uint32_t g_totalEpochs = 0;
	int32_t g_globalStep = 0;
	uint32_t g_batchIdx = 0;
	uint32_t g_epochBatchTotal = 0; ///< batch count from last completed epoch
	bool g_fitStarted = false;

	/* =========================================================================
	 * JSON helpers
	 * ====================================================================== */

	std::string jsonStr(const std::string& json, const std::string& key) {
		const std::string search = "\"" + key + "\":\"";
		const auto pos = json.find(search);
		if (pos == std::string::npos)
			return {};
		const auto start = pos + search.size();
		const auto end = json.find('"', start);
		if (end == std::string::npos)
			return {};
		return json.substr(start, end - start);
	}

	int64_t jsonInt(const std::string& json, const std::string& key) {
		const std::string search = "\"" + key + "\":";
		const auto pos = json.find(search);
		if (pos == std::string::npos)
			return 0;
		auto p = pos + search.size();
		bool neg = false;
		if (p < json.size() && json[p] == '-') {
			neg = true;
			++p;
		}
		int64_t v = 0;
		for (; p < json.size() && json[p] >= '0' && json[p] <= '9'; ++p)
			v = v * 10 + static_cast<int64_t>(json[p] - '0');
		return neg ? -v : v;
	}

	std::string stripAnsi(const std::string& s) {
		std::string out;
		out.reserve(s.size());
		bool inEsc = false;
		for (char c : s) {
			if (c == '\033') {
				inEsc = true;
				continue;
			}
			if (inEsc) {
				if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
					inEsc = false;
				continue;
			}
			if (c == '\n' || c == '\r')
				continue;
			out += c;
		}
		return out;
	}

	/* =========================================================================
	 * Shared DOM builders — used by both rendering modes
	 * ====================================================================== */

	using namespace ftxui;

	Element makeMetricsElement() {
		Elements rows;
		for (const auto& key : g_metricKeys) {
			const auto it = g_metrics.find(key);
			if (it == g_metrics.end())
				continue;
			const MetricHistory& h = it->second;
			if (h.values.empty())
				continue;

			const float latest = h.values.back();
			const float vmin = h.vmin;
			const float vmax = h.vmax;

			auto spark =
					graph([vals = std::vector<float>(h.values.begin(), h.values.end()), vmin, vmax](int w, int ht) {
						std::vector<int> out(static_cast<size_t>(w), 0);
						const int count = static_cast<int>(vals.size());
						const float range = vmax - vmin;
						for (int i = 0; i < w; ++i) {
							const int src = count - w + i;
							if (src < 0 || src >= count)
								continue;
							float norm = (range > 1e-9f) ? (vals[static_cast<size_t>(src)] - vmin) / range : 0.5f;
							norm = std::clamp(norm, 0.0f, 1.0f);
							out[static_cast<size_t>(i)] = static_cast<int>(norm * static_cast<float>(ht));
						}
						return out;
					}) |
					color(Color::GreenLight) | flex;

			char buf[24]{};
			if (latest >= 1e3f || (latest != 0.0f && latest < 1e-3f))
				snprintf(buf, sizeof(buf), "%.3e", static_cast<double>(latest));
			else
				snprintf(buf, sizeof(buf), "%.4f", static_cast<double>(latest));

			rows.push_back(
					hbox({
							text(key) | color(Color::Cyan) | size(WIDTH, EQUAL, 16),
							text(" "),
							spark,
							text(" "),
							text(buf) | color(Color::Yellow) | size(WIDTH, EQUAL, 10),
					}) |
					size(HEIGHT, EQUAL, 3) | flex
			);
		}
		if (rows.empty())
			rows.push_back(text("  No metrics yet — waiting for training to start…") | dim);
		return vbox(std::move(rows));
	}

	Element makeLogsElement() {
		Elements lines;
		const int total = static_cast<int>(g_logs.size());
		const int start = std::max(0, total - kMaxLogs);
		for (int i = start; i < total; ++i) {
			const LogEntry& e = g_logs[static_cast<size_t>(i)];
			Element prefix;
			switch (e.level) {
			case TMM_LOG_TRACE:
				prefix = text("[TRACE] ") | color(Color::GrayDark);
				break;
			case TMM_LOG_DEBUG:
				prefix = text("[DEBUG] ") | color(Color::Blue);
				break;
			case TMM_LOG_INFO:
				prefix = text("[INFO]  ") | color(Color::Green);
				break;
			case TMM_LOG_WARN:
				prefix = text("[WARN]  ") | color(Color::Yellow);
				break;
			case TMM_LOG_ERROR:
				prefix = text("[ERROR] ") | color(Color::Red) | bold;
				break;
			default:
				prefix = text("        ");
				break;
			}
			lines.push_back(hbox({prefix, text(stripAnsi(e.message))}));
		}
		if (lines.empty())
			lines.push_back(text("  No log messages yet…") | dim);
		return vbox(std::move(lines)) | frame;
	}

} // anonymous namespace

/* =========================================================================
 * DOM-only rendering (WASM / non-interactive native)
 * ====================================================================== */

namespace {

#ifdef __EMSCRIPTEN__
	__attribute__((import_module("tmm"), import_name("tmm_terminal_size"))) extern void
	tmm_terminal_size(uint32_t* out_width, uint32_t* out_height);
#endif

	void redrawDom() {
		uint32_t w = 80, h = 24;
#ifdef __EMSCRIPTEN__
		tmm_terminal_size(&w, &h);
#endif
		w = std::max(w, 20u);
		h = std::max(h, 8u);
		const int cols = static_cast<int>(w);
		const int rows = static_cast<int>(h);

		const std::string epochStr =
				g_fitStarted ? (std::to_string(g_currentEpoch) + "/" + std::to_string(g_totalEpochs)) : "—";

		auto header = window(
				text(" TMM Training ") | bold,
				vbox({
						hbox({text("Model:   ") | bold,
							  text(g_modelPath.empty() ? "—" : g_modelPath) | color(Color::Cyan) | flex, separator(),
							  text(" Epoch: ") | bold, text(epochStr) | color(Color::Yellow)}),
						hbox({text("Dataset: ") | bold,
							  text(g_datasetUri.empty() ? "—" : g_datasetUri) | color(Color::Cyan) | flex, separator(),
							  text(" Split: ") | bold, text(g_datasetSplit.empty() ? "—" : g_datasetSplit)}),
						hbox({text("Step:    ") | bold, text(std::to_string(g_globalStep)) | color(Color::Green),
							  filler()}),
				})
		);

		auto metricsPanel = window(text(" Metrics ") | bold, makeMetricsElement()) | flex;
		auto logsPanel = window(text(" Logs ") | bold, makeLogsElement() | flex) | flex;

		auto doc = vbox({header, metricsPanel, logsPanel}) | size(HEIGHT, EQUAL, rows) | size(WIDTH, EQUAL, cols);

		Screen screen(cols, rows);
		Render(screen, doc);
		fputs("\033[H", stdout);
		fputs(screen.ToString().c_str(), stdout);
		fflush(stdout);
	}

} // anonymous namespace

/* =========================================================================
 * Interactive mode — native only
 * ====================================================================== */

#ifndef __EMSCRIPTEN__
namespace interactive {

	static std::mutex g_mtx;
	static std::thread g_thread;
	static ScreenInteractive* g_screen = nullptr;
	static int g_tab = 0;	 ///< 0=Metrics, 1=Logs
	static int g_infoW = 36; ///< resizable info pane width

	/* -----------------------------------------------------------------------
	 * RAM utilisation
	 * -------------------------------------------------------------------- */

#ifdef __linux__
	struct MemInfo {
		float fraction;
		std::string label;
	};
	static MemInfo queryMem() {
		struct rusage ru{};
		getrusage(RUSAGE_SELF, &ru);
		struct sysinfo si{};
		sysinfo(&si);
		const uint64_t total = static_cast<uint64_t>(si.totalram) * si.mem_unit;
		const uint64_t rss = static_cast<uint64_t>(ru.ru_maxrss) * 1024; // kb→bytes
		const float frac = total > 0 ? static_cast<float>(rss) / static_cast<float>(total) : 0.0f;
		char buf[64];
		snprintf(
				buf, sizeof(buf), "%llu MB / %llu MB", (unsigned long long)(rss / (1024 * 1024)),
				(unsigned long long)(total / (1024 * 1024))
		);
		return {frac, buf};
	}
#else
	struct MemInfo {
		float fraction;
		std::string label;
	};
	static MemInfo queryMem() { return {0.0f, "N/A"}; }
#endif

	/* -----------------------------------------------------------------------
	 * Component renderers
	 * -------------------------------------------------------------------- */

	static Component makeUtilizationComponent() {
		return Renderer([] {
			const auto mem = queryMem();
			return window(
					text(" Utilization ") | bold,
					hbox({
							text("RAM: ") | color(Color::Cyan),
							gaugeRight(mem.fraction) | color(LinearGradient(Color::Green, Color::Red)) | flex,
							text("  " + mem.label) | color(Color::GrayDark),
					})
			);
		});
	}

	static Component makeInfoComponent() {
		return Renderer([] {
			std::lock_guard<std::mutex> lk(g_mtx);
			const std::string epochStr =
					g_fitStarted ? (std::to_string(g_currentEpoch) + "/" + std::to_string(g_totalEpochs)) : "—";

			Elements rows;
			auto row = [&](const std::string& k, const std::string& v) {
				rows.push_back(hbox({
						text(k) | color(Color::Cyan),
						filler(),
						text(v) | color(Color::GrayLight),
				}));
			};
			if (!g_modelPath.empty())
				row("Model", g_modelPath);
			if (!g_datasetUri.empty())
				row("Dataset", g_datasetUri);
			if (!g_datasetSplit.empty())
				row("Split", g_datasetSplit);
			if (g_fitStarted)
				row("Epoch", epochStr);
			row("Step", std::to_string(g_globalStep));

			if (rows.empty())
				rows.push_back(text("—") | dim);
			return window(text(" Info ") | bold, vbox(std::move(rows))) | flex;
		});
	}

	static Component makeTabComponent() {
		static std::vector<std::string> kTabLabels = {"  Metrics  ", "  Logs  "};
		auto toggle = Toggle(&kTabLabels, &g_tab);

		auto metricsRenderer = Renderer([] {
			std::lock_guard<std::mutex> lk(g_mtx);
			return makeMetricsElement();
		});
		auto logsRenderer = Renderer([] {
			std::lock_guard<std::mutex> lk(g_mtx);
			return makeLogsElement();
		});
		auto tabContent = Container::Tab({metricsRenderer, logsRenderer}, &g_tab);

		auto container = Container::Vertical({toggle, tabContent | flex});
		return Renderer(container, [=] {
			return window(hbox({toggle->Render()}), tabContent->Render() | flex) | flex;
		});
	}

	static Component makeProgressComponent() {
		return Renderer([] {
			std::lock_guard<std::mutex> lk(g_mtx);
			const float progress =
					(g_epochBatchTotal > 0)
							? std::clamp(
									  static_cast<float>(g_batchIdx) / static_cast<float>(g_epochBatchTotal), 0.0f, 1.0f
							  )
							: 0.0f;
			const std::string epochStr =
					g_fitStarted ? (std::to_string(g_currentEpoch) + "/" + std::to_string(g_totalEpochs)) : "—";
			char pct[16];
			snprintf(pct, sizeof(pct), " %5.1f%%", static_cast<double>(progress) * 100.0);
			return window(
					text(" Progress ") | bold,
					hbox({
							text("Epoch " + epochStr + "  ") | color(Color::Yellow),
							gaugeRight(progress) | color(LinearGradient(Color::Cyan, Color::Blue)) | flex,
							text(pct),
					})
			);
		});
	}

	/* -----------------------------------------------------------------------
	 * Render thread entry point
	 * -------------------------------------------------------------------- */

	static void runLoop() {
		auto screen = ScreenInteractive::Fullscreen();
		g_screen = &screen;

		auto utilComp = makeUtilizationComponent();
		auto tabComp = makeTabComponent();
		auto infoComp = makeInfoComponent();
		auto progComp = makeProgressComponent();

		auto centreLeft = Container::Vertical({tabComp | flex});
		auto split = ResizableSplitRight(infoComp, centreLeft, &g_infoW);
		auto root = Container::Vertical({
				utilComp,
				split | flex,
				progComp,
		});

		/* Quit on 'q' or Ctrl-C (Ctrl-C is handled by SIGINT; q for convenience). */
		auto withQuit = CatchEvent(root, [&](Event ev) {
			if (ev == Event::Character('q') || ev == Event::Escape) {
				screen.Exit();
				return true;
			}
			return false;
		});

		screen.Loop(withQuit);
		g_screen = nullptr;
	}

	static void postRedraw() {
		if (g_screen != nullptr)
			g_screen->PostEvent(Event::Custom);
	}

} // namespace interactive
#endif // !__EMSCRIPTEN__

/* =========================================================================
 * Plugin globals — mode selection
 * ====================================================================== */

namespace {
	bool g_interactive = false;
}

/* =========================================================================
 * ABI exports
 * ====================================================================== */

extern "C" {

static tmm_plugin_info g_info = {
		TMM_ABI_VERSION, "console-ui", "0.2.0", "Live TUI training dashboard — FTXUI, WASI/native"
};

tmm_plugin_info* tmm_plugin_get_info(void) { return &g_info; }

tmm_error tmm_plugin_init(const tmm_host_api* /*host*/, const char* cfg, uint32_t len) {
	/* Parse "interactive" option from config JSON (native only). */
#ifndef __EMSCRIPTEN__
	if (cfg != nullptr && len > 0) {
		const std::string cfgStr(cfg, len);
		/* Simple check: "interactive":true */
		g_interactive =
				(cfgStr.find("\"interactive\":true") != std::string::npos ||
				 cfgStr.find("\"interactive\": true") != std::string::npos);
	}
	if (g_interactive) {
		interactive::g_thread = std::thread(interactive::runLoop);
		return TMM_OK;
	}
#endif
	fputs("\033[2J\033[H", stdout);
	fflush(stdout);
	redrawDom();
	return TMM_OK;
}

void tmm_plugin_teardown(void) {
#ifndef __EMSCRIPTEN__
	if (g_interactive) {
		if (interactive::g_screen != nullptr)
			interactive::g_screen->Exit();
		if (interactive::g_thread.joinable())
			interactive::g_thread.join();
		g_interactive = false;
		interactive::g_screen = nullptr;
		return;
	}
#endif
	uint32_t w = 80, h = 24;
#ifdef __EMSCRIPTEN__
	tmm_terminal_size(&w, &h);
#endif
	char buf[32]{};
	snprintf(buf, sizeof(buf), "\033[%u;0H\n", h);
	fputs(buf, stdout);
	fflush(stdout);
}

/* -----------------------------------------------------------------------
 * Lifecycle callbacks
 * -------------------------------------------------------------------- */

void tmm_on_fit_begin(const char* ctx_json, uint32_t len) {
	const std::string json(ctx_json, len);
#ifndef __EMSCRIPTEN__
	if (g_interactive) {
		std::lock_guard<std::mutex> lk(interactive::g_mtx);
		g_fitStarted = true;
		g_currentEpoch = 0;
		g_globalStep = 0;
		g_datasetUri = jsonStr(json, "dataset_uri");
		g_datasetSplit = jsonStr(json, "dataset_split");
		g_modelPath = jsonStr(json, "model_path");
		g_totalEpochs = static_cast<uint32_t>(jsonInt(json, "total_epochs"));
		interactive::postRedraw();
		return;
	}
#endif
	g_fitStarted = true;
	g_currentEpoch = 0;
	g_globalStep = 0;
	g_datasetUri = jsonStr(json, "dataset_uri");
	g_datasetSplit = jsonStr(json, "dataset_split");
	g_modelPath = jsonStr(json, "model_path");
	g_totalEpochs = static_cast<uint32_t>(jsonInt(json, "total_epochs"));
	redrawDom();
}

void tmm_on_epoch_begin(uint32_t epoch, uint32_t /*total*/) {
#ifndef __EMSCRIPTEN__
	if (g_interactive) {
		std::lock_guard<std::mutex> lk(interactive::g_mtx);
		g_currentEpoch = epoch;
		g_batchIdx = 0;
		interactive::postRedraw();
		return;
	}
#endif
	g_currentEpoch = epoch;
	g_batchIdx = 0;
	redrawDom();
}

void tmm_on_batch_end(
		uint32_t batch, float /*loss*/, const char* /*json*/, uint32_t /*len*/
) {
#ifndef __EMSCRIPTEN__
	if (g_interactive) {
		std::lock_guard<std::mutex> lk(interactive::g_mtx);
		g_batchIdx = batch + 1;
		interactive::postRedraw();
		return;
	}
#endif
	static uint32_t s_count = 0;
	if ((++s_count % 10u) == 0u)
		redrawDom();
}

int32_t tmm_on_epoch_end(uint32_t epoch, const char* /*json*/, uint32_t /*len*/) {
#ifndef __EMSCRIPTEN__
	if (g_interactive) {
		std::lock_guard<std::mutex> lk(interactive::g_mtx);
		g_currentEpoch = epoch;
		g_epochBatchTotal = g_batchIdx; // freeze total for next epoch's progress bar
		interactive::postRedraw();
		return 0;
	}
#endif
	g_epochBatchTotal = g_batchIdx;
	g_currentEpoch = epoch;
	redrawDom();
	return 0;
}

void tmm_on_fit_end(const char* /*json*/, uint32_t /*len*/) {
#ifndef __EMSCRIPTEN__
	if (g_interactive) {
		interactive::postRedraw();
		return;
	}
#endif
	redrawDom();
}

void tmm_on_log(uint32_t level, const char* msg, uint32_t len) {
#ifndef __EMSCRIPTEN__
	if (g_interactive) {
		std::lock_guard<std::mutex> lk(interactive::g_mtx);
		if (static_cast<int>(g_logs.size()) >= kMaxLogs)
			g_logs.pop_front();
		g_logs.push_back({level, std::string(msg, len)});
		interactive::postRedraw();
		return;
	}
#endif
	if (static_cast<int>(g_logs.size()) >= kMaxLogs)
		g_logs.pop_front();
	g_logs.push_back({level, std::string(msg, len)});
	redrawDom();
}

void tmm_on_metric(const char* key, uint32_t key_len, float value, int32_t step) {
#ifndef __EMSCRIPTEN__
	if (g_interactive) {
		std::lock_guard<std::mutex> lk(interactive::g_mtx);
		g_globalStep = step;
		const std::string k(key, key_len);
		if (g_metrics.find(k) == g_metrics.end())
			g_metricKeys.push_back(k);
		g_metrics[k].push(value);
		interactive::postRedraw();
		return;
	}
#endif
	g_globalStep = step;
	const std::string k(key, key_len);
	if (g_metrics.find(k) == g_metrics.end())
		g_metricKeys.push_back(k);
	g_metrics[k].push(value);
	redrawDom();
}

} // extern "C"

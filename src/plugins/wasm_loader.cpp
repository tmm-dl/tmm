/**
 * @file wasm_loader.cpp
 * @brief WAMR-based plugin loader implementation.
 *
 * @details
 * All WAMR API usage is confined to this translation unit so that the rest
 * of the plugin library does not pull in WAMR headers.
 */

#include "wasm_loader.hpp"

#include "plugin_ctx.hpp"

#include <ttm/plugins/abi.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <ttm/compat/format.hpp>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <ttm/compat/expected.hpp>
#include <vector>

#include <lib_export.h>
#include <wasm_export.h>

namespace ttm::plugins {

	/* =========================================================================
	 * Runtime init / destroy — refcounted so multiple PluginManager instances
	 * (e.g. in unit tests) are safe.
	 * ====================================================================== */

	namespace {
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) -- module-level refcount is intentionally mutable; no alternative without dynamic allocation or thread_local
		std::atomic<int> g_wamr_refcount{0};
	} // namespace

	std::expected<void, std::string> wasm_loader_init() {
		if (g_wamr_refcount.fetch_add(1, std::memory_order_acq_rel) == 0) {
			RuntimeInitArgs args{};
			args.mem_alloc_type = Alloc_With_System_Allocator;

			if (!wasm_runtime_full_init(
						&args
				)) { // NOLINT(readability-implicit-bool-conversion) -- WAMR API returns bool-like int
				g_wamr_refcount.fetch_sub(1, std::memory_order_acq_rel);
				return std::unexpected("wasm_loader_init: wasm_runtime_full_init failed");
			}
		}
		return {};
	}

	void wasm_loader_destroy() {
		if (g_wamr_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			wasm_runtime_destroy();
		}
	}

	/* =========================================================================
	 * String helpers
	 * ====================================================================== */

	std::expected<void, std::string>
	// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- out_wasm_ptr and out_len are output parameters with distinct semantics (WASM address vs byte count); names make this clear
	wasm_push_string(wasm_module_inst_t inst, std::string_view str, uint32_t& out_wasm_ptr, uint32_t& out_len) {
		const auto len = static_cast<uint32_t>(str.size());
		void* native_ptr = nullptr;
		const uint32_t wasm_ptr = wasm_runtime_module_malloc(inst, len, &native_ptr);
		if (wasm_ptr == 0) {
			return std::unexpected(
					"wasm_push_string: failed to allocate " + std::to_string(len) + " bytes in WASM linear memory"
			);
		}
		std::memcpy(native_ptr, str.data(), len);
		out_wasm_ptr = wasm_ptr;
		out_len = len;
		return {};
	}

	/* =========================================================================
	 * WASM adapter types
	 *
	 * These wrap plugin-side function-table indices so that host code can call
	 * back into a WASM plugin through the standard IDatasetSource / ITask
	 * C++ interfaces.
	 * ====================================================================== */

	namespace {

		/**
		 * @brief Pack an i64 value into two consecutive uint32_t argv slots.
		 * @param[out] lo  Low 32 bits.
		 * @param[out] hi  High 32 bits.
		 */
		static void pack_i64(uint32_t& lo, uint32_t& hi, int64_t val) {
			const auto u = static_cast<uint64_t>(val);
			lo = static_cast<uint32_t>(u & 0xFFFF'FFFFu);
			hi = static_cast<uint32_t>(u >> 32u);
		}

		/** @brief Unpack an i64 from two consecutive uint32_t argv slots. */
		static int64_t unpack_i64(uint32_t lo, uint32_t hi) {
			return static_cast<int64_t>((static_cast<uint64_t>(hi) << 32u) | static_cast<uint64_t>(lo));
		}

		/**
		 * @brief Call a WASM function-table entry and return void.
		 * @tparam N  Number of 32-bit argument slots.
		 */
		template <std::size_t N>
		static void call_indirect_void(wasm_exec_env_t env, uint32_t elem_idx, std::array<uint32_t, N>& argv) {
			wasm_runtime_call_indirect(
					env, elem_idx, static_cast<uint32_t>(N), argv.data()
			); // NOLINT(readability-implicit-bool-conversion) -- WAMR bool-like return ignored; errors surfaced via exceptions / return values
		}

		/**
		 * @brief Call a WASM function-table entry and return the i32 result in argv[0].
		 */
		template <std::size_t N>
		static uint32_t call_indirect_i32(wasm_exec_env_t env, uint32_t elem_idx, std::array<uint32_t, N>& argv) {
			wasm_runtime_call_indirect(env, elem_idx, static_cast<uint32_t>(N), argv.data());
			return argv[0];
		}

		/**
		 * @brief Call a WASM function-table entry and return the i64 result (argv[0..1]).
		 */
		template <std::size_t N>
		static int64_t call_indirect_i64(wasm_exec_env_t env, uint32_t elem_idx, std::array<uint32_t, N>& argv) {
			wasm_runtime_call_indirect(env, elem_idx, static_cast<uint32_t>(N), argv.data());
			return unpack_i64(argv[0], argv[1]);
		}

		/**
		 * @brief Read a NUL-terminated C string from WASM linear memory into std::string.
		 * @return Empty string if wasm_ptr is 0 or the address is invalid.
		 */
		static std::string read_wasm_cstr(wasm_module_inst_t inst, uint32_t wasm_ptr) {
			if (wasm_ptr == 0) {
				return {};
			}
			const auto* native = static_cast<const char*>(wasm_runtime_addr_app_to_native(inst, wasm_ptr));
			if (native == nullptr) {
				return {};
			}
			return std::string(native);
		}

		/**
		 * @brief Walk a WASM null-terminated i32[] array of string pointers.
		 * @return Vector of host-side std::string copies.
		 */
		static std::vector<std::string> read_wasm_string_array(wasm_module_inst_t inst, uint32_t arr_wasm_ptr) {
			std::vector<std::string> result;
			if (arr_wasm_ptr == 0) {
				return result;
			}
			const auto* arr = static_cast<const uint32_t*>(wasm_runtime_addr_app_to_native(inst, arr_wasm_ptr));
			if (arr == nullptr) {
				return result;
			}
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- null-terminated WASM array; bounds are checked by the null sentinel
			for (const uint32_t* p = arr; *p != 0; ++p) {
				result.push_back(read_wasm_cstr(inst, *p));
			}
			return result;
		}

		/* -----------------------------------------------------------------
		 * WasmByteReader
		 * ---------------------------------------------------------------- */

		/**
		 * @brief IByteReader that calls back into a WASM plugin via function-table indices.
		 *
		 * @details
		 * Buffers for read/seek operations are allocated in WASM linear memory for
		 * each call and freed immediately after, to avoid accumulating heap usage.
		 */
		class WasmByteReader final : public IByteReader {
		public:
			/**
			 * @param[in] env       WASM execution environment (owned by Plugin).
			 * @param[in] inst      WASM module instance (owned by Plugin).
			 * @param[in] handle    ttm_handle returned by the source's open() function.
			 * @param[in] readIdx   Function-table index of the read() function.
			 * @param[in] seekIdx   Function-table index of the seek() function (0 = not exported).
			 * @param[in] closeIdx  Function-table index of the close() function.
			 */
			WasmByteReader(
					wasm_exec_env_t env, wasm_module_inst_t inst, ttm_handle handle, uint32_t readIdx, uint32_t seekIdx,
					uint32_t closeIdx
			)
					: env(env), inst(inst), handle(handle), readIdx(readIdx), seekIdx(seekIdx), closeIdx(closeIdx) {}

			WasmByteReader(const WasmByteReader&) = delete;
			WasmByteReader& operator=(const WasmByteReader&) = delete;
			WasmByteReader(WasmByteReader&&) = delete;
			WasmByteReader& operator=(WasmByteReader&&) = delete;

			~WasmByteReader() override {
				if (handle != TTM_INVALID_HANDLE && closeIdx != 0) {
					std::array<uint32_t, 2> argv{};
					pack_i64(argv[0], argv[1], handle);
					call_indirect_void(env, closeIdx, argv);
				}
			}

			std::streamsize read(std::byte* hostBuf, std::streamsize n) override {
				if (readIdx == 0 || n <= 0) {
					return 0;
				}
				/* Allocate a WASM-side buffer */
				void* wasmNative = nullptr;
				const uint32_t wasmBuf = wasm_runtime_module_malloc(inst, static_cast<uint32_t>(n), &wasmNative);
				if (wasmBuf == 0) {
					return -1;
				}

				/* argv: [handle_lo, handle_hi, buf_wasm_ptr, len] */
				std::array<uint32_t, 4> argv{};
				pack_i64(argv[0], argv[1], handle);
				argv[2] = wasmBuf;
				argv[3] = static_cast<uint32_t>(n);
				const auto nRead = static_cast<std::streamsize>(call_indirect_i32(env, readIdx, argv));

				/* Copy result to host buffer */
				if (nRead > 0 && wasmNative != nullptr) {
					std::memcpy(hostBuf, wasmNative, static_cast<std::size_t>(nRead));
				}
				wasm_runtime_module_free(inst, wasmBuf);
				return nRead;
			}

			[[nodiscard]] bool seekable() const noexcept override { return seekIdx != 0; }

			std::streampos seek(std::streamoff off, std::ios_base::seekdir dir) override {
				if (seekIdx == 0) {
					return {std::streamoff{-1}};
				}
				int32_t whence = 0;
				switch (dir) {
				case std::ios_base::beg:
					whence = 0;
					break;
				case std::ios_base::cur:
					whence = 1;
					break;
				case std::ios_base::end:
					whence = 2;
					break;
				default:
					return {std::streamoff{-1}};
				}
				/* argv: [handle_lo, handle_hi, off_lo, off_hi, whence] */
				/* Return: i64 in argv[0..1] */
				std::array<uint32_t, 5> argv{};
				pack_i64(argv[0], argv[1], handle);
				pack_i64(argv[2], argv[3], static_cast<int64_t>(off));
				argv[4] = static_cast<uint32_t>(whence);
				const auto pos = call_indirect_i64(env, seekIdx, argv);
				return (pos < 0) ? std::streampos{std::streamoff{-1}}
								 : std::streampos{static_cast<std::streamoff>(pos)};
			}

		private:
			wasm_exec_env_t env;
			wasm_module_inst_t inst;
			ttm_handle handle;
			uint32_t readIdx;
			uint32_t seekIdx;
			uint32_t closeIdx;
		};

		/* -----------------------------------------------------------------
		 * WasmSourceAdapter
		 * ---------------------------------------------------------------- */

		/**
		 * @brief IDatasetSource backed by a WASM plugin's ttm_source_vtable.
		 *
		 * @details
		 * Holds function-table indices extracted from the vtable struct in WASM
		 * linear memory.  All WASM calls are made through the env/inst pair that
		 * is owned by the plugin's Plugin record, which lives as long as the
		 * PluginManager.
		 */
		class WasmSourceAdapter final : public IDatasetSource {
		public:
			/**
			 * @param[in] schemeList  Schemes this source handles.
			 * @param[in] env         WASM execution environment (owned by Plugin).
			 * @param[in] inst        WASM module instance (owned by Plugin).
			 * @param[in] openIdx     Function-table index for open().
			 * @param[in] readIdx     Function-table index for read().
			 * @param[in] seekIdx     Function-table index for seek() (0 = not exported).
			 * @param[in] closeIdx    Function-table index for close().
			 */
			WasmSourceAdapter(
					std::vector<std::string> schemeList, wasm_exec_env_t env, wasm_module_inst_t inst, uint32_t openIdx,
					uint32_t readIdx, uint32_t seekIdx, uint32_t closeIdx
			)
					: schemeList(std::move(schemeList)), env(env), inst(inst), openIdx(openIdx), readIdx(readIdx),
					  seekIdx(seekIdx), closeIdx(closeIdx) {}

			[[nodiscard]] std::vector<std::string> schemes() const override { return schemeList; }

			std::unique_ptr<IByteReader> open(std::string_view uri) override {
				if (openIdx == 0) {
					return nullptr;
				}
				/* Push URI into WASM memory */
				uint32_t uriWasmPtr = 0;
				uint32_t uriLen = 0;
				if (!wasm_push_string(inst, uri, uriWasmPtr, uriLen)) {
					return nullptr;
				}

				/* Allocate error buffer in WASM memory */
				constexpr uint32_t kErrBufCap = 256;
				void* errNative = nullptr;
				const uint32_t errWasmPtr = wasm_runtime_module_malloc(inst, kErrBufCap, &errNative);

				ttm_handle handle = TTM_INVALID_HANDLE;
				if (errWasmPtr != 0) {
					/* argv: [uri_wasm_ptr, uri_len, err_wasm_ptr, err_cap] — returns i64 */
					std::array<uint32_t, 4> argv{uriWasmPtr, uriLen, errWasmPtr, kErrBufCap};
					handle = call_indirect_i64(env, openIdx, argv);

					if (handle == TTM_INVALID_HANDLE && errNative != nullptr) {
						std::cerr << "[ttm] WasmSourceAdapter::open error: " << static_cast<const char*>(errNative)
								  << '\n';
					}
					wasm_runtime_module_free(inst, errWasmPtr);
				}
				wasm_runtime_module_free(inst, uriWasmPtr);

				if (handle == TTM_INVALID_HANDLE) {
					return nullptr;
				}
				return std::make_unique<WasmByteReader>(env, inst, handle, readIdx, seekIdx, closeIdx);
			}

		private:
			std::vector<std::string> schemeList;
			wasm_exec_env_t env;
			wasm_module_inst_t inst;
			uint32_t openIdx;
			uint32_t readIdx;
			uint32_t seekIdx;
			uint32_t closeIdx;
		};

		/* -----------------------------------------------------------------
		 * WasmTaskAdapter
		 * ---------------------------------------------------------------- */

		/**
		 * @brief ITask backed by a WASM plugin's ttm_task_vtable.
		 *
		 * @details
		 * Task vtable functions return pointers to WASM static strings.  We call
		 * them once at construction time and cache the results as std::string so
		 * that the ITask interface returns stable string_view values.
		 */
		class WasmTaskAdapter final : public ITask {
		public:
			/**
			 * @brief Construct by calling each vtable function on the WASM plugin.
			 *
			 * @param[in] env      WASM execution environment.
			 * @param[in] inst     WASM module instance.
			 * @param[in] nameFn   Function-table index for name().
			 * @param[in] aliasesFn    Function-table index for aliases().
			 * @param[in] inputsFn     Function-table index for input_features().
			 * @param[in] labelFn      Function-table index for label_feature().
			 * @param[in] metricsFn    Function-table index for default_metrics().
			 */
			WasmTaskAdapter(
					wasm_exec_env_t env, wasm_module_inst_t inst, uint32_t nameFn, uint32_t aliasesFn,
					uint32_t inputsFn, uint32_t labelFn, uint32_t metricsFn
			) {
				auto callStr = [&](uint32_t fn) -> std::string {
					if (fn == 0) {
						return {};
					}
					std::array<uint32_t, 1> argv{};
					call_indirect_i32(env, fn, argv);
					return read_wasm_cstr(inst, argv[0]);
				};

				auto callStrArr = [&](uint32_t fn) -> std::vector<std::string> {
					if (fn == 0) {
						return {};
					}
					std::array<uint32_t, 1> argv{};
					call_indirect_i32(env, fn, argv);
					return read_wasm_string_array(inst, argv[0]);
				};

				nameStr = callStr(nameFn);
				labelStr = callStr(labelFn);
				aliasStrs = callStrArr(aliasesFn);
				inputStrs = callStrArr(inputsFn);
				metricStrs = callStrArr(metricsFn);
			}

			[[nodiscard]] std::string_view name() const override { return nameStr; }

			[[nodiscard]] std::vector<std::string_view> aliases() const override {
				std::vector<std::string_view> v;
				v.reserve(aliasStrs.size());
				for (const auto& s : aliasStrs) {
					v.emplace_back(s);
				}
				return v;
			}

			[[nodiscard]] std::vector<std::string_view> input_features() const override {
				std::vector<std::string_view> v;
				v.reserve(inputStrs.size());
				for (const auto& s : inputStrs) {
					v.emplace_back(s);
				}
				return v;
			}

			[[nodiscard]] std::string_view label_feature() const override { return labelStr; }

			[[nodiscard]] std::vector<std::string_view> default_metrics() const override {
				std::vector<std::string_view> v;
				v.reserve(metricStrs.size());
				for (const auto& s : metricStrs) {
					v.emplace_back(s);
				}
				return v;
			}

		private:
			std::string nameStr;
			std::string labelStr;
			std::vector<std::string> aliasStrs;
			std::vector<std::string> inputStrs;
			std::vector<std::string> metricStrs;
		};

		/**
		 * @brief Retrieve the Plugin* stored as WAMR user-data.
		 *
		 * @details
		 * Replaces the previous pattern of storing `ttm_host_api*` as user-data,
		 * which caused a use-after-free when the stack-local ttm_host_api in
		 * PluginManager::load() was deallocated while the plugin env still held a
		 * pointer to it.  Storing `Plugin*` instead is safe because Plugin lives
		 * in PluginManager::plugins (heap, stable address) for the process lifetime.
		 */
		static Plugin* get_plugin_ud(wasm_exec_env_t env) {
			return static_cast<Plugin*>(wasm_runtime_get_user_data(env));
		}

		/* ---------- log --------------------------------------------------------- */
		// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- signature is fixed by WAMR NativeSymbol ABI; parameter names clearly distinguish them
		void host_log(wasm_exec_env_t env, uint32_t level, uint32_t msg_ptr, uint32_t msg_len) {
			auto* inst = wasm_runtime_get_module_inst(env);
			const auto* msg = static_cast<const char*>(wasm_runtime_addr_app_to_native(inst, msg_ptr));
			if (msg == nullptr) return;
			const auto* p = get_plugin_ud(env);
			if (p != nullptr && p->persistentApi.log != nullptr) {
				p->persistentApi.log(p->persistentApi.ctx, static_cast<ttm_log_level>(level), msg, msg_len);
			}
		}

		/* ---------- log_metric -------------------------------------------------- */
		/* Route the metric back through persistentApi.log_metric (→ s_log_metric →
		 * PluginManager::emit_metric).  This avoids a direct dependency on the full
		 * PluginManager type, which is only forward-declared in wasm_loader.hpp. */
		// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- fixed WAMR signature; all parameters have distinct types and clear names
		void host_log_metric(wasm_exec_env_t env, uint32_t key_ptr, uint32_t key_len, float value, int32_t step) {
			auto* inst = wasm_runtime_get_module_inst(env);
			const auto* key = static_cast<const char*>(wasm_runtime_addr_app_to_native(inst, key_ptr));
			if (key == nullptr || key_len == 0) return;
			const auto* p = get_plugin_ud(env);
			if (p != nullptr && p->persistentApi.log_metric != nullptr) {
				p->persistentApi.log_metric(p->persistentApi.ctx, key, key_len, value, step);
			}
		}

		/* ---------- terminal_size ----------------------------------------------- */
		void host_terminal_size(wasm_exec_env_t env, uint32_t w_ptr, uint32_t h_ptr) {
			auto* inst = wasm_runtime_get_module_inst(env);
			auto* w = static_cast<uint32_t*>(wasm_runtime_addr_app_to_native(inst, w_ptr));
			auto* h = static_cast<uint32_t*>(wasm_runtime_addr_app_to_native(inst, h_ptr));
			uint32_t width = 80, height = 24;
#if defined(TIOCGWINSZ)
			struct winsize ws{};
			// NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg) -- TIOCGWINSZ ioctl is the standard POSIX way to get terminal size; no safer alternative
			if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
				if (ws.ws_col > 0) width  = static_cast<uint32_t>(ws.ws_col);
				if (ws.ws_row > 0) height = static_cast<uint32_t>(ws.ws_row);
			}
#endif
			if (w != nullptr) *w = width;
			if (h != nullptr) *h = height;
		}

		/* ---------- alloc ------------------------------------------------------- */
		uint32_t host_alloc(wasm_exec_env_t env, uint32_t size) {
			auto* inst = wasm_runtime_get_module_inst(env);
			void* native_ptr = nullptr;
			return wasm_runtime_module_malloc(inst, size, &native_ptr);
		}

		/* ---------- free -------------------------------------------------------- */
		void host_free(wasm_exec_env_t env, uint32_t wasm_ptr) {
			auto* inst = wasm_runtime_get_module_inst(env);
			wasm_runtime_module_free(inst, wasm_ptr);
		}

		/* ---------- register_source -------------------------------------------- */
		int32_t host_register_source(wasm_exec_env_t env, uint32_t schemes_ptr, uint32_t vtable_ptr) {
			auto* inst = wasm_runtime_get_module_inst(env);
			if (schemes_ptr == 0 || vtable_ptr == 0) return TTM_ERR_ARGS;

			/* Read scheme list from WASM memory */
			auto schemeList = read_wasm_string_array(inst, schemes_ptr);
			if (schemeList.empty()) return TTM_ERR_ARGS;

			/* Read vtable function-table indices — [open_idx, read_idx, seek_idx, close_idx] */
			const auto* vtNative = static_cast<const uint32_t*>(wasm_runtime_addr_app_to_native(inst, vtable_ptr));
			if (vtNative == nullptr) return TTM_ERR_ARGS;
			uint32_t openIdx = 0, readIdx = 0, seekIdx = 0, closeIdx = 0;
			std::memcpy(&openIdx,  vtNative + 0, sizeof(uint32_t));
			std::memcpy(&readIdx,  vtNative + 1, sizeof(uint32_t));
			std::memcpy(&seekIdx,  vtNative + 2, sizeof(uint32_t));
			std::memcpy(&closeIdx, vtNative + 3, sizeof(uint32_t));

			/* Registration context is valid only during ttm_plugin_init */
			const auto* p = get_plugin_ud(env);
			if (p == nullptr) return TTM_ERR_ARGS;
			auto* regCtx = static_cast<PluginRegistrationCtx*>(p->persistentApi.ctx);
			if (regCtx == nullptr || !regCtx->attach_source) return TTM_ERR_ARGS;

			auto adapter = std::make_unique<WasmSourceAdapter>(
					std::move(schemeList), env, inst, openIdx, readIdx, seekIdx, closeIdx
			);
			regCtx->attach_source(std::move(adapter));
			return TTM_OK;
		}

		/* ---------- register_task ---------------------------------------------- */
		int32_t host_register_task(wasm_exec_env_t env, uint32_t name_ptr, uint32_t vtable_ptr) {
			auto* inst = wasm_runtime_get_module_inst(env);
			if (name_ptr == 0 || vtable_ptr == 0) return TTM_ERR_ARGS;

			/* Read vtable function-table indices (6 × uint32_t) */
			const auto* vtNative = static_cast<const uint32_t*>(wasm_runtime_addr_app_to_native(inst, vtable_ptr));
			if (vtNative == nullptr) return TTM_ERR_ARGS;
			uint32_t nameFn = 0, aliasesFn = 0, inputsFn = 0, labelFn = 0, metricsFn = 0, lossFn = 0;
			std::memcpy(&nameFn,    vtNative + 0, sizeof(uint32_t));
			std::memcpy(&aliasesFn, vtNative + 1, sizeof(uint32_t));
			std::memcpy(&inputsFn,  vtNative + 2, sizeof(uint32_t));
			std::memcpy(&labelFn,   vtNative + 3, sizeof(uint32_t));
			std::memcpy(&metricsFn, vtNative + 4, sizeof(uint32_t));
			std::memcpy(&lossFn,    vtNative + 5, sizeof(uint32_t));

			const auto* p = get_plugin_ud(env);
			if (p == nullptr) return TTM_ERR_ARGS;
			auto* regCtx = static_cast<PluginRegistrationCtx*>(p->persistentApi.ctx);
			if (regCtx == nullptr || !regCtx->attach_task) return TTM_ERR_ARGS;

			auto adapter = std::make_unique<WasmTaskAdapter>(
					env, inst, nameFn, aliasesFn, inputsFn, labelFn, metricsFn
			);
			if (adapter->name().empty()) return TTM_ERR_ARGS;

			regCtx->attach_task(std::move(adapter));
			return TTM_OK;
		}

		/* ---------- NativeSymbol table ----------------------------------------- */
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables) -- WAMR requires a mutable NativeSymbol[] passed to wasm_runtime_register_natives
		NativeSymbol ttm_native_symbols[] = {
				/* { "export_name", func_ptr, "signature", attachment } */
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- WAMR NativeSymbol API requires void* function pointers; no safer alternative
				{"ttm_log",             reinterpret_cast<void*>(host_log),            "(iii)",  nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
				{"ttm_log_metric",      reinterpret_cast<void*>(host_log_metric),     "(iifi)", nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
				{"ttm_terminal_size",   reinterpret_cast<void*>(host_terminal_size),  "(ii)",   nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
				{"ttm_alloc",           reinterpret_cast<void*>(host_alloc),          "(i)i",   nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
				{"ttm_free",            reinterpret_cast<void*>(host_free),           "(i)",    nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
				{"ttm_register_source", reinterpret_cast<void*>(host_register_source),"(ii)i",  nullptr},
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- see above
				{"ttm_register_task",   reinterpret_cast<void*>(host_register_task),  "(ii)i",  nullptr},
		};

	} // anonymous namespace

	/* =========================================================================
	 * Load / unload
	 * ====================================================================== */

	std::expected<void, std::string> wasm_loader_load(
			const std::filesystem::path& path, std::string_view config_json, const ttm_host_api& host_api,
			Plugin& plugin
	) {
		/* 1. Read file bytes ------------------------------------------------- */
		std::ifstream file(path, std::ios::binary);
		if (!file) {
			return std::unexpected("wasm_loader_load: cannot open '" + path.string() + "'");
		}
		std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

		/* 2. Register host symbols ------------------------------------------- */
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay,cppcoreguidelines-pro-bounds-constant-array-index) -- WAMR API requires array-to-pointer decay for NativeSymbol table
		if (!wasm_runtime_register_natives( // NOLINT(readability-implicit-bool-conversion) -- WAMR API returns bool-like int
					"ttm", ttm_native_symbols, sizeof(ttm_native_symbols) / sizeof(ttm_native_symbols[0]) // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
				)) {
			return std::unexpected("wasm_loader_load: failed to register host symbols");
		}

		/* 3. Load (compile) module ------------------------------------------- */
		constexpr uint32_t kErrBufSize = 256;
		std::array<char, kErrBufSize> errorbuf{};
		plugin.module =
				wasm_runtime_load(bytes.data(), static_cast<uint32_t>(bytes.size()), errorbuf.data(), errorbuf.size());
		if (plugin.module == nullptr) {
			return std::unexpected(std::format("wasm_loader_load: wasm_runtime_load failed: {}", errorbuf.data()));
		}

		/* 4. Instantiate ----------------------------------------------------- */
		constexpr uint32_t STACK_SIZE = 512 * 1024;          /*  512 KB */
		constexpr uint32_t HEAP_SIZE  = 32 * 1024 * 1024;   /* 32 MB — FTXUI 6.x needs room for std::deque/unordered_map */
		plugin.inst = wasm_runtime_instantiate(plugin.module, STACK_SIZE, HEAP_SIZE, errorbuf.data(), errorbuf.size());
		if (plugin.inst == nullptr) {
			wasm_runtime_unload(plugin.module);
			plugin.module = nullptr;
			return std::unexpected(std::format("wasm_loader_load: instantiate failed: {}", errorbuf.data()));
		}

		/* 5. Create execution environment ------------------------------------ */
		plugin.env = wasm_runtime_create_exec_env(plugin.inst, STACK_SIZE);
		if (plugin.env == nullptr) {
			wasm_runtime_deinstantiate(plugin.inst);
			wasm_runtime_unload(plugin.module);
			plugin.inst = nullptr;
			plugin.module = nullptr;
			return std::unexpected("wasm_loader_load: failed to create exec env");
		}

		/* Copy host_api into Plugin so callbacks can access it safely after
		 * PluginManager::load() returns and the stack-local PluginRegistrationCtx
		 * goes out of scope.  Extract the owning PluginManager via the ctx pointer
		 * (valid here because we are still inside wasm_loader_load).
		 *
		 * ctx is set to nullptr now; PluginManager::load() updates it to manager*
		 * after a successful load so that post-init callbacks (log_metric, etc.)
		 * receive a valid PluginManager* context. */
		plugin.persistentApi = host_api;
		if (const auto* regCtx = static_cast<const PluginRegistrationCtx*>(host_api.ctx)) {
			plugin.manager = regCtx->manager;
		}
		plugin.persistentApi.ctx = nullptr; /* updated to manager* by PluginManager::load() */
		/* Store Plugin* as WAMR user-data — stable: Plugin lives on the heap
		 * inside PluginManager::plugins for the entire manager lifetime. */
		wasm_runtime_set_user_data(plugin.env, &plugin);

		/* 6. Verify ABI version ---------------------------------------------- */
		auto* fn_info = wasm_runtime_lookup_function(plugin.inst, "ttm_plugin_get_info");
		if (fn_info == nullptr) {
			wasm_loader_unload(plugin);
			return std::unexpected("wasm_loader_load: '" + path.string() + "' does not export ttm_plugin_get_info");
		}
		std::array<uint32_t, 1> info_args{};
		if (!wasm_runtime_call_wasm(plugin.env, fn_info, 0, info_args.data())) {
			wasm_loader_unload(plugin);
			return std::unexpected(
					"wasm_loader_load: ttm_plugin_get_info call failed: " +
					std::string(wasm_runtime_get_exception(plugin.inst))
			);
		}
		const auto* info =
				static_cast<const ttm_plugin_info*>(wasm_runtime_addr_app_to_native(plugin.inst, info_args[0]));
		if (info == nullptr) {
			wasm_loader_unload(plugin);
			return std::unexpected("wasm_loader_load: ttm_plugin_get_info returned null");
		}
		if (info->abiVersion != TTM_ABI_VERSION) {
			wasm_loader_unload(plugin);
			return std::unexpected(
					"wasm_loader_load: plugin ABI version mismatch (plugin=" + std::to_string(info->abiVersion) +
					", host=" + std::to_string(TTM_ABI_VERSION) + ")"
			);
		}

		/* 7. Call ttm_plugin_init -------------------------------------------- */
		auto* fn_init = wasm_runtime_lookup_function(plugin.inst, "ttm_plugin_init");
		if (fn_init == nullptr) {
			wasm_loader_unload(plugin);
			return std::unexpected("wasm_loader_load: '" + path.string() + "' does not export ttm_plugin_init");
		}

		/* Push config JSON into WASM linear memory */
		uint32_t config_wasm_ptr = 0;
		uint32_t config_len = 0;
		if (auto r = wasm_push_string(plugin.inst, config_json, config_wasm_ptr, config_len); !r) {
			wasm_loader_unload(plugin);
			return std::unexpected(r.error());
		}

		/* Push host_api struct into WASM linear memory */
		void* api_native_ptr = nullptr;
		const uint32_t api_wasm_ptr = wasm_runtime_module_malloc(plugin.inst, sizeof(ttm_host_api), &api_native_ptr);
		if (api_wasm_ptr == 0) {
			wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
			wasm_loader_unload(plugin);
			return std::unexpected("wasm_loader_load: failed to allocate host_api in WASM heap");
		}
		std::memcpy(api_native_ptr, &host_api, sizeof(ttm_host_api));

		/* init(host_api_ptr, config_ptr, config_len) → i32 */
		std::array<uint32_t, 3> init_args{api_wasm_ptr, config_wasm_ptr, config_len};
		if (!wasm_runtime_call_wasm(plugin.env, fn_init, init_args.size(), init_args.data())) {
			wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
			wasm_runtime_module_free(plugin.inst, api_wasm_ptr);
			wasm_loader_unload(plugin);
			return std::unexpected(
					"wasm_loader_load: ttm_plugin_init failed: " + std::string(wasm_runtime_get_exception(plugin.inst))
			);
		}
		const auto init_result = static_cast<ttm_error>(init_args[0]);
		if (init_result != TTM_OK) {
			wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
			wasm_runtime_module_free(plugin.inst, api_wasm_ptr);
			wasm_loader_unload(plugin);
			return std::unexpected(
					"wasm_loader_load: ttm_plugin_init returned error " + std::to_string(static_cast<int>(init_result))
			);
		}

		/* Free temporaries in WASM heap */
		wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
		wasm_runtime_module_free(plugin.inst, api_wasm_ptr);

		/* 8. Resolve optional lifecycle hooks -------------------------------- */
		auto lookup = [&](const char* name) -> wasm_function_inst_t {
			return wasm_runtime_lookup_function(plugin.inst, name);
		};
		plugin.fnFitBegin      = lookup("ttm_on_fit_begin");
		plugin.fnEpochBegin    = lookup("ttm_on_epoch_begin");
		plugin.fnBatchBegin    = lookup("ttm_on_batch_begin");
		plugin.fnLossComputed  = lookup("ttm_on_loss_computed");
		plugin.fnBatchEnd      = lookup("ttm_on_batch_end");
		plugin.fnEpochEnd      = lookup("ttm_on_epoch_end");
		plugin.fnValidationEnd = lookup("ttm_on_validation_end");
		plugin.fnFitEnd        = lookup("ttm_on_fit_end");
		plugin.fnOnLog         = lookup("ttm_on_log");
		plugin.fnOnMetric      = lookup("ttm_on_metric");
		plugin.fnTeardown      = lookup("ttm_plugin_teardown");

		return {};
	}

	void wasm_loader_unload(Plugin& plugin) {
		if (plugin.env != nullptr && plugin.fnTeardown != nullptr) {
			wasm_runtime_call_wasm(plugin.env, plugin.fnTeardown, 0, nullptr);
		}
		if (plugin.env != nullptr) {
			wasm_runtime_destroy_exec_env(plugin.env);
			plugin.env = nullptr;
		}
		if (plugin.inst != nullptr) {
			wasm_runtime_deinstantiate(plugin.inst);
			plugin.inst = nullptr;
		}
		if (plugin.module != nullptr) {
			wasm_runtime_unload(plugin.module);
			plugin.module = nullptr;
		}
	}

} // namespace ttm::plugins

/**
 * @file plugin_manager.cpp
 * @brief PluginManager implementation.
 *
 * @details
 * This file contains:
 * - The Plugin struct definition (declared opaque in plugin_manager.hpp).
 * - The built-in local filesystem source (FileSource / FileReader).
 * - CSourceAdapter — wraps a C ttm_source_vtable into an IDatasetSource.
 * - All PluginManager member function definitions.
 */

#include <ttm/plugins/plugin_manager.hpp>

#include "wasm_loader.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ttm::plugins {

	/* =========================================================================
	 * PluginManager::Plugin definition
	 * ====================================================================== */

	/**
	 * @brief Full definition of the opaque Plugin struct.
	 *
	 * @details
	 * Declared in plugin_manager.hpp as an incomplete type; defined here so that
	 * wasm_loader.hpp types are not exposed in the public header.
	 *
	 * @see wasm_loader.hpp  Origin of the Plugin type
	 */
	struct PluginManager::Plugin : ::ttm::plugins::Plugin {};

	/* =========================================================================
	 * Built-in local filesystem source
	 * ====================================================================== */

	namespace {

		/**
		 * @brief IByteReader backed by a local file (std::ifstream).
		 * @details Registered automatically for the "file:" URI scheme.
		 */
		class FileReader final : public IByteReader {
		public:
			/**
			 * @param[in] path  Absolute or relative filesystem path to open.
			 */
			explicit FileReader(const std::string& path) : file(path, std::ios::binary) {}

			[[nodiscard]] bool valid() const { return static_cast<bool>(file); }

			std::streamsize read(std::byte* buf, std::streamsize n) override {
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- necessary: std::ifstream::read() takes char* but IByteReader::read() takes std::byte*; both are single-byte types
				file.read(reinterpret_cast<char*>(buf), n);
				return file.gcount();
			}

			bool seekable() const noexcept override { return true; }

			std::streampos seek(std::streamoff off, std::ios_base::seekdir dir) override {
				file.seekg(off, dir);
				if (!file) {
					return {std::streamoff{-1}};
				}
				return file.tellg();
			}

		private:
			std::ifstream file;
		};

		/**
		 * @brief IDatasetSource for "file:" URIs.
		 *
		 * @details
		 * Strips the "file://" prefix (or "file:" for bare paths) and opens the
		 * remainder as a local filesystem path.
		 *
		 * @par Supported URI forms
		 * - `file:///absolute/path/to/file.arrow`
		 * - `file://relative/path.arrow`
		 * - `file:relative/path.arrow`
		 */
		class FileSource final : public IDatasetSource {
		public:
			[[nodiscard]] std::vector<std::string> schemes() const override { return {"file:"}; }

			std::unique_ptr<IByteReader> open(std::string_view uri) override {
				/* Strip "file://" or "file:" prefix */
				constexpr std::size_t kFileDoubleSlashPfxLen = 7; // "file://"
				constexpr std::size_t kFileSinglePfxLen = 5;	  // "file:"
				std::string path{uri};
				if (path.substr(0, kFileDoubleSlashPfxLen) == "file://") {
					path = path.substr(kFileDoubleSlashPfxLen);
				} else if (path.substr(0, kFileSinglePfxLen) == "file:") {
					path = path.substr(kFileSinglePfxLen);
				}

				auto reader = std::make_unique<FileReader>(path);
				if (!reader->valid()) {
					return nullptr;
				}
				return reader;
			}
		};

		/* =========================================================================
		 * CSourceAdapter — wraps a C ttm_source_vtable into IDatasetSource
		 * ====================================================================== */

		/**
		 * @brief IByteReader backed by a plugin-provided C ttm_source_vtable handle.
		 */
		class CVtableReader final : public IByteReader {
		public:
			/**
			 * @param[in] vt      Source vtable provided by the plugin.
			 * @param[in] handle  Handle returned by vt->open().
			 */
			CVtableReader(const ttm_source_vtable& vt, ttm_handle handle) : vt(vt), handle(handle) {}

			CVtableReader(const CVtableReader&) = delete;
			CVtableReader& operator=(const CVtableReader&) = delete;
			CVtableReader(CVtableReader&&) = delete;
			CVtableReader& operator=(CVtableReader&&) = delete;

			~CVtableReader() override {
				if (handle != TTM_INVALID_HANDLE && vt.close != nullptr) {
					vt.close(handle);
				}
			}

			std::streamsize read(std::byte* buf, std::streamsize n) override {
				if (vt.read == nullptr) {
					return -1;
				}
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- C ABI passes void*; buf is std::byte* which is ABI-compatible
				return static_cast<std::streamsize>(vt.read(handle, buf, static_cast<int32_t>(n)));
			}

			[[nodiscard]] bool seekable() const noexcept override { return vt.seek != nullptr; }

			std::streampos seek(std::streamoff off, std::ios_base::seekdir dir) override {
				if (vt.seek == nullptr) {
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
				const auto pos = vt.seek(handle, static_cast<int64_t>(off), whence);
				return (pos < 0) ? std::streampos{std::streamoff{-1}} : std::streampos{pos};
			}

		private:
			const ttm_source_vtable vt;
			ttm_handle handle;
		};

		/**
		 * @brief IDatasetSource that wraps a plugin's ttm_source_vtable.
		 *
		 * @details
		 * Created by PluginManager::s_register_source() and stored in the owning
		 * Plugin's sources list.
		 */
		class CSourceAdapter final : public IDatasetSource {
		public:
			/**
			 * @param[in] schemeList  Schemes this source handles (copied).
			 * @param[in] vt          Source vtable (must remain valid for this object's lifetime).
			 */
			CSourceAdapter(std::vector<std::string> schemeList, const ttm_source_vtable* vt)
					: schemeList(std::move(schemeList)), vt(*vt) {}

			[[nodiscard]] std::vector<std::string> schemes() const override { return schemeList; }

			std::unique_ptr<IByteReader> open(std::string_view uri) override {
				constexpr std::size_t kErrBufSize = 256;
				// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- C API requires a char[] error buffer; std::array cannot be passed to C vtable open()
				char errBuf[kErrBufSize] = {};
				// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- required by C vtable API
				const auto handle = vt.open(uri.data(), static_cast<uint32_t>(uri.size()), errBuf, kErrBufSize);

				if (handle == TTM_INVALID_HANDLE) {
					std::cerr << "[ttm] CSourceAdapter::open error: " << errBuf << '\n'; // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- errBuf is a C error message buffer
					return nullptr;
				}
				return std::make_unique<CVtableReader>(vt, handle);
			}

		private:
			std::vector<std::string> schemeList;
			ttm_source_vtable vt; /* Copy of the vtable struct */
		};

	} // anonymous namespace

	/* =========================================================================
	 * PluginManager — named constructor
	 * ====================================================================== */

	std::expected<PluginManager, std::string> PluginManager::create() {
		PluginManager mgr;

		if (auto r = wasm_loader_init(); !r) {
			return std::unexpected(r.error());
		}
		mgr.wamrRefOwned = true;

		/* Register the built-in local filesystem source */
		auto fileSrc = std::make_unique<FileSource>();
		for (const auto& scheme : fileSrc->schemes()) {
			mgr.sourceRegistry.emplace(scheme, fileSrc.get());
		}
		/* Store a dummy Plugin record to own the built-in source */
		auto builtinPlugin = std::make_unique<Plugin>();
		builtinPlugin->sources.push_back(std::move(fileSrc));
		mgr.plugins.push_back(std::move(builtinPlugin));

		return mgr;
	}

	/* =========================================================================
 * PluginManager — destructor and move operations
 * ====================================================================== */

	PluginManager::~PluginManager() {
		if (!wamrRefOwned) {
			return;
		}

		/* Unload all plugins (index 0 is the built-in, no WASM handles to release) */
		for (auto& p : plugins) {
			wasm_loader_unload(*p);
		}
		plugins.clear();
		sourceRegistry.clear();

		wasm_loader_destroy();
	}

	PluginManager::PluginManager(PluginManager&& other) noexcept
			: plugins(std::move(other.plugins)), sourceRegistry(std::move(other.sourceRegistry)),
			  wamrRefOwned(other.wamrRefOwned) {
		other.wamrRefOwned = false;
	}

	PluginManager& PluginManager::operator=(PluginManager&& other) noexcept {
		if (this != &other) {
			/* Destroy current state */
			this->~PluginManager();
			/* Move from other */
			plugins = std::move(other.plugins);
			sourceRegistry = std::move(other.sourceRegistry);
			wamrRefOwned = other.wamrRefOwned;
			other.wamrRefOwned = false;
		}
		return *this;
	}

	/* =========================================================================
 * Plugin loading
 * ====================================================================== */

	std::expected<void, std::string>
	PluginManager::load(const std::filesystem::path& path, std::string_view config_json) {
		auto p = std::make_unique<Plugin>();
		auto hostApi = make_host_api();

		if (auto r = wasm_loader_load(path, config_json, hostApi, *p); !r) {
			return std::unexpected(r.error());
		}

		plugins.push_back(std::move(p));
		return {};
	}

	/* =========================================================================
 * Source registry
 * ====================================================================== */

	IDatasetSource* PluginManager::find_source(std::string_view scheme) const {
		const auto it = sourceRegistry.find(std::string(scheme));
		return (it != sourceRegistry.end()) ? it->second : nullptr;
	}

	/* =========================================================================
 * Host API construction
 * ====================================================================== */

	ttm_host_api PluginManager::make_host_api() {
		ttm_host_api api{};
		api.ctx = this;
		api.register_source = &PluginManager::s_register_source;
		api.register_transform = &PluginManager::s_register_transform;
		api.register_task = &PluginManager::s_register_task;
		api.register_metric = &PluginManager::s_register_metric;
		api.log = &PluginManager::s_log;
		api.alloc = &PluginManager::s_alloc;
		api.free = &PluginManager::s_free;
		return api;
	}

	/* =========================================================================
 * Host API static callbacks
 * ====================================================================== */

	ttm_error PluginManager::s_register_source(void* ctx, const char** schemes, const ttm_source_vtable* vt) {
		if (ctx == nullptr || schemes == nullptr || vt == nullptr) {
			return TTM_ERR_ARGS;
		}

		std::vector<std::string> schemeList;
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- C ABI requires null-terminated pointer array; no safer alternative without copying
		for (const char** sch = schemes; *sch != nullptr; ++sch) {
			schemeList.emplace_back(*sch);
		}
		if (schemeList.empty()) {
			return TTM_ERR_ARGS;
		}

		auto* self = static_cast<PluginManager*>(ctx);
		auto adapter = std::make_unique<CSourceAdapter>(std::move(schemeList), vt);

		/* The plugin currently being loaded is always the last one pushed before
     * the init call; we pass nullptr here and let register_source_impl
     * attach the source to the most-recently-added plugin. */
		self->register_source_impl(std::move(adapter), nullptr);
		return TTM_OK;
	}

	void PluginManager::
			register_source_impl(std::unique_ptr<IDatasetSource> src, Plugin* /*owner — reserved for future use*/) {
		for (const auto& scheme : src->schemes()) {
			if (sourceRegistry.contains(scheme)) {
				std::cerr << "[ttm] warning: source scheme '" << scheme << "' already registered; overriding.\n";
			}
			sourceRegistry[scheme] = src.get();
		}
		/* Attach to the last plugin (the one currently being initialised) */
		assert(!plugins.empty());
		plugins.back()->sources.push_back(std::move(src));
	}

	ttm_error PluginManager::s_register_transform(
			void* /*ctx*/, const char* /*name*/, const char** /*aliases*/, const ttm_transform_vtable* /*vt*/
	) {
		/* TODO: implement transform registry */
		return TTM_ERR_UNSUPPORTED;
	}

	ttm_error PluginManager::s_register_task(
			void* /*ctx*/, const char* /*name*/, const char** /*aliases*/, const ttm_task_vtable* /*vt*/
	) {
		/* TODO: implement task registry */
		return TTM_ERR_UNSUPPORTED;
	}

	ttm_error PluginManager::s_register_metric(
			void* /*ctx*/, const char* /*name*/, const char** /*aliases*/, const ttm_metric_vtable* /*vt*/
	) {
		/* TODO: implement metric registry */
		return TTM_ERR_UNSUPPORTED;
	}

	void PluginManager::s_log(void* /*ctx*/, ttm_log_level level, const char* msg, uint32_t len) {
		// NOLINTNEXTLINE(cppcoreguidelines-init-variables) -- always set by the switch below; initializing here would generate a clang-analyzer-deadcode.DeadStores warning
		const char* prefix;
		switch (level) {
		case TTM_LOG_TRACE:
			prefix = "[TRACE] ";
			break;
		case TTM_LOG_DEBUG:
			prefix = "[DEBUG] ";
			break;
		case TTM_LOG_INFO:
			prefix = "[INFO]  ";
			break;
		case TTM_LOG_WARN:
			prefix = "[WARN]  ";
			break;
		case TTM_LOG_ERROR:
			prefix = "[ERROR] ";
			break;
		default:
			prefix = "";
			break;
		}
		std::cerr << "[ttm plugin] " << prefix;
		std::cerr.write(msg, static_cast<std::streamsize>(len));
		std::cerr << '\n';
	}

	// NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory) -- host alloc/free callbacks must use malloc/free for ABI compatibility with WASM plugin heap allocations
	void* PluginManager::s_alloc(void* /*ctx*/, uint32_t size) { return std::malloc(size); }

	// NOLINTNEXTLINE(cppcoreguidelines-no-malloc,hicpp-no-malloc,cppcoreguidelines-owning-memory) -- see s_alloc above
	void PluginManager::s_free(void* /*ctx*/, void* ptr) { std::free(ptr); }

	/* =========================================================================
 * Lifecycle event dispatch
 *
 * Each emit_* function iterates plugins and calls the resolved hook on
 * plugins that exported it.  String arguments are pushed into WASM linear
 * memory for each call and freed immediately after.
 * ====================================================================== */

	/** @brief Helper — push a string to WASM memory, call fn, then free. */
	static void call_with_string(Plugin& plug, wasm_function_inst_t fn, std::string_view str) {
		if (fn == nullptr) {
			return;
		}
		uint32_t wasmPtr = 0;
		uint32_t wasmLen = 0;
		if (!wasm_push_string(plug.inst, str, wasmPtr, wasmLen)) {
			return;
		}
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- WAMR call ABI requires C array for args
		uint32_t args[2] = {wasmPtr, wasmLen};
		wasm_runtime_call_wasm(plug.env, fn, 2, args); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- WAMR ABI
		wasm_runtime_module_free(plug.inst, wasmPtr);
	}

	void PluginManager::emit_fit_begin(std::string_view ctx_json) {
		for (auto& p : plugins) {
			call_with_string(*p, p->fnFitBegin, ctx_json);
		}
	}

	void PluginManager::emit_epoch_begin(std::uint32_t epoch, std::uint32_t total) {
		for (auto& p : plugins) {
			if (p->fnEpochBegin == nullptr) {
				continue;
			}
			// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- WAMR call ABI requires C array for args
			uint32_t args[2] = {epoch, total};
			wasm_runtime_call_wasm(p->env, p->fnEpochBegin, 2, args); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- WAMR ABI
		}
	}

	void PluginManager::emit_batch_begin(std::uint32_t batch, std::uint32_t total) {
		for (auto& p : plugins) {
			if (p->fnBatchBegin == nullptr) {
				continue;
			}
			// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- WAMR call ABI requires C array for args
			uint32_t args[2] = {batch, total};
			wasm_runtime_call_wasm(p->env, p->fnBatchBegin, 2, args); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- WAMR ABI
		}
	}

	float PluginManager::emit_loss_computed(float loss) {
		for (auto& p : plugins) {
			if (p->fnLossComputed == nullptr) {
				continue;
			}
			/* WAMR passes f32 as uint32 bit-cast */
			// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- WAMR call ABI requires C array for args
			uint32_t args[1] = {};
			std::memcpy(&args[0], &loss, sizeof(float));
			wasm_runtime_call_wasm(p->env, p->fnLossComputed, 1, args); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- WAMR ABI
			std::memcpy(&loss, &args[0], sizeof(float));
		}
		return loss;
	}

	// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- batch(uint32_t) and loss(float) are different types; semantics are clear from names
	void PluginManager::emit_batch_end(std::uint32_t batch, float loss, std::string_view metrics_json) {
		for (auto& p : plugins) {
			if (p->fnBatchEnd == nullptr) {
				continue;
			}
			uint32_t metricsPtr = 0;
			uint32_t metricsLen = 0;
			if (!wasm_push_string(p->inst, metrics_json, metricsPtr, metricsLen)) {
				continue;
			}
			uint32_t lossBits = 0;
			std::memcpy(&lossBits, &loss, sizeof(float));
			// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- WAMR call ABI requires C array for args
			uint32_t args[4] = {batch, lossBits, metricsPtr, metricsLen};
			wasm_runtime_call_wasm(p->env, p->fnBatchEnd, 4, args); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- WAMR ABI
			wasm_runtime_module_free(p->inst, metricsPtr);
		}
	}

	bool PluginManager::emit_epoch_end(std::uint32_t epoch, std::string_view metrics_json) {
		bool stop = false;
		for (auto& p : plugins) {
			if (p->fnEpochEnd == nullptr) {
				continue;
			}
			uint32_t metricsPtr = 0;
			uint32_t metricsLen = 0;
			if (!wasm_push_string(p->inst, metrics_json, metricsPtr, metricsLen)) {
				continue;
			}
			// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- WAMR call ABI requires C array for args
			uint32_t args[3] = {epoch, metricsPtr, metricsLen};
			wasm_runtime_call_wasm(p->env, p->fnEpochEnd, 3, args); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- WAMR ABI
			wasm_runtime_module_free(p->inst, metricsPtr);
			if (args[0] != 0) {
				stop = true;
			}
		}
		return stop;
	}

	void PluginManager::emit_validation_end(std::string_view metrics_json) {
		for (auto& p : plugins) {
			call_with_string(*p, p->fnValidationEnd, metrics_json);
		}
	}

	void PluginManager::emit_fit_end(std::string_view metrics_json) {
		for (auto& p : plugins) {
			call_with_string(*p, p->fnFitEnd, metrics_json);
		}
	}

} // namespace ttm::plugins

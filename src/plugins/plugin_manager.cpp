/**
 * @file plugin_manager.cpp
 * @brief PluginManager implementation.
 *
 * @details
 * This file contains:
 * - The Plugin struct definition (declared opaque in plugin_manager.hpp).
 * - The built-in local filesystem source (FileSource / FileReader).
 * - CSourceAdapter — wraps a C ttm_source_vtable into an IDatasetSource.
 * - CTaskAdapter   — wraps a C ttm_task_vtable into an ITask.
 * - All PluginManager member function definitions.
 */

#include <ttm/plugins/plugin_manager.hpp>

#include "native_loader.hpp"
#include "plugin_ctx.hpp"
#include "wasm_loader.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <ttm/compat/expected.hpp>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#	include <sys/ioctl.h>
#	include <unistd.h>
#endif

#include <wasm_export.h>

#include <ttm/plugins/abi.h>
#include <ttm/plugins/extension.hpp>

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

	/**
	 * @brief Full definition of the opaque NativePlugin struct.
	 * @see native_loader.hpp
	 */
	struct PluginManager::NativePlugin : ::ttm::plugins::NativePlugin {};

	/**
	 * @brief Full definition of PluginRegistrationCtx — mirrors plugin_ctx.hpp
	 * but uses the manager-internal derived types.
	 */
	struct PluginManager::PluginRegistrationCtx : ::ttm::plugins::PluginRegistrationCtx {};

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
				std::array<char, kErrBufSize> errBuf{};
				const auto handle =
						vt.open(uri.data(), static_cast<uint32_t>(uri.size()), errBuf.data(), errBuf.size());

				if (handle == TTM_INVALID_HANDLE) {
					std::cerr << "[ttm] CSourceAdapter::open error: " << errBuf.data() << '\n';
					return nullptr;
				}
				return std::make_unique<CVtableReader>(vt, handle);
			}

		private:
			std::vector<std::string> schemeList;
			ttm_source_vtable vt; /* Copy of the vtable struct */
		};

		/* =========================================================================
		 * CTaskAdapter — wraps a C ttm_task_vtable into ITask
		 * ====================================================================== */

		/**
		 * @brief Helper — walk a NULL-terminated const char** array into a vector.
		 */
		static std::vector<std::string_view> walk_string_array(const char** arr) {
			std::vector<std::string_view> result;
			if (arr == nullptr) {
				return result;
			}
			// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- C ABI null-terminated array; no safer alternative
			for (const char** p = arr; *p != nullptr; ++p) {
				result.emplace_back(*p);
			}
			return result;
		}

		/**
		 * @brief ITask that wraps a plugin's ttm_task_vtable.
		 *
		 * @details
		 * The vtable function pointers return pointers into plugin-owned static
		 * memory (valid for the lifetime of the plugin).  We cache the results as
		 * std::string so that ITask callers receive stable std::string_view values
		 * even if the plugin is later unloaded and reloaded.
		 */
		class CTaskAdapter final : public ITask {
		public:
			explicit CTaskAdapter(const ttm_task_vtable& vt) : vt(vt) {
				if (vt.name != nullptr) {
					nameStr = vt.name();
				}
				if (vt.label_feature != nullptr) {
					labelStr = vt.label_feature();
				}
				if (vt.aliases != nullptr) {
					for (auto sv : walk_string_array(vt.aliases())) {
						aliasStrs.emplace_back(sv);
					}
				}
				if (vt.input_features != nullptr) {
					for (auto sv : walk_string_array(vt.input_features())) {
						inputStrs.emplace_back(sv);
					}
				}
				if (vt.default_metrics != nullptr) {
					for (auto sv : walk_string_array(vt.default_metrics())) {
						metricStrs.emplace_back(sv);
					}
				}
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
			ttm_task_vtable vt;
			std::string nameStr;
			std::string labelStr;
			std::vector<std::string> aliasStrs;
			std::vector<std::string> inputStrs;
			std::vector<std::string> metricStrs;
		};

	/* =========================================================================
	 * CModelLoaderAdapter — wraps a C ttm_model_loader_vtable into IModelLoader
	 * ====================================================================== */

	/**
	 * @brief IModelLoader that delegates all calls through a plugin vtable.
	 *
	 * @details
	 * Created by PluginManager::s_register_model_loader() and stored in the
	 * owning NativePlugin's modelLoaders list.
	 */
	class CModelLoaderAdapter final : public IModelLoader {
	public:
		explicit CModelLoaderAdapter(const ttm_model_loader_vtable* vt) : vt_(*vt) {}

		[[nodiscard]] bool probe(std::string_view path) const override {
			if (vt_.probe == nullptr) return false;
			return vt_.probe(path.data(), static_cast<uint32_t>(path.size())) != 0;
		}

		[[nodiscard]] std::expected<ttm_handle, std::string>
		open(std::string_view path, std::string_view cfg_json) override {
			if (vt_.load == nullptr) return std::unexpected("model loader vtable has no load()");
			constexpr std::size_t kErrBufSize = 512;
			std::array<char, kErrBufSize> errBuf{};
			const auto h = vt_.load(
				path.data(),     static_cast<uint32_t>(path.size()),
				cfg_json.data(), static_cast<uint32_t>(cfg_json.size()),
				errBuf.data(), kErrBufSize
			);
			if (h == TTM_INVALID_HANDLE) {
				return std::unexpected(errBuf[0] != '\0'
					? std::string(errBuf.data())
					: "model load failed (no error message)");
			}
			return h;
		}

		[[nodiscard]] ttm_model_info_t get_info(ttm_handle h) const override {
			if (vt_.get_info == nullptr) return ttm_model_info_t{};
			return vt_.get_info(h);
		}

		ttm_error describe_params(
			ttm_handle h,
			const ttm_param_desc_t** out_descs,
			uint32_t* out_count
		) override {
			if (vt_.describe_params == nullptr) {
				if (out_count != nullptr) *out_count = 0;
				return TTM_ERR_UNSUPPORTED;
			}
			return vt_.describe_params(h, out_descs, out_count);
		}

		ttm_error bind_params(
			ttm_handle h,
			const DLTensor* params, uint32_t param_count,
			const DLTensor* grads,  uint32_t grad_count
		) override {
			if (vt_.bind_params == nullptr) return TTM_ERR_UNSUPPORTED;
			return vt_.bind_params(h, params, param_count, grads, grad_count);
		}

		ttm_error init_params(ttm_handle h, std::string_view method) override {
			if (vt_.init_params == nullptr) return TTM_OK;
			return vt_.init_params(h, method.data(), static_cast<uint32_t>(method.size()));
		}

		ttm_error step(
			ttm_handle h,
			const DLTensor* inputs, uint32_t n,
			float* out_loss
		) override {
			if (vt_.step == nullptr) return TTM_ERR_UNSUPPORTED;
			return vt_.step(h, inputs, n, out_loss);
		}

		ttm_error infer(
			ttm_handle h,
			const DLTensor* inputs,  uint32_t in_count,
			DLTensor*       outputs, uint32_t* out_count
		) override {
			if (vt_.infer == nullptr) return TTM_ERR_UNSUPPORTED;
			return vt_.infer(h, inputs, in_count, outputs, out_count);
		}

		ttm_error zero_grad(ttm_handle h) override {
			if (vt_.zero_grad == nullptr) return TTM_OK;
			return vt_.zero_grad(h);
		}

		void destroy(ttm_handle h) override {
			if (vt_.destroy != nullptr) vt_.destroy(h);
		}

	private:
		ttm_model_loader_vtable vt_;
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

		/* Unload native plugins first */
		for (auto& p : nativePlugins) {
			native_loader_unload(*p);
		}
		nativePlugins.clear();

		/* Unload WASM plugins (index 0 is the built-in, no WASM handles to release) */
		for (auto& p : plugins) {
			wasm_loader_unload(*p);
		}
		plugins.clear();

		sourceRegistry.clear();
		taskRegistry.clear();
		modelLoaderRegistry.clear();
		transformRegistry.clear();

		wasm_loader_destroy();
	}

	PluginManager::PluginManager(PluginManager&& other) noexcept
			: plugins(std::move(other.plugins)), nativePlugins(std::move(other.nativePlugins)),
			  sourceRegistry(std::move(other.sourceRegistry)),
			  taskRegistry(std::move(other.taskRegistry)),
			  modelLoaderRegistry(std::move(other.modelLoaderRegistry)),
			  transformRegistry(std::move(other.transformRegistry)),
			  schedulerVtableRegistry(std::move(other.schedulerVtableRegistry)),
			  wamrRefOwned(other.wamrRefOwned) {
		other.wamrRefOwned = false;
	}

	PluginManager& PluginManager::operator=(PluginManager&& other) noexcept {
		if (this != &other) {
			/* Destroy current state */
			this->~PluginManager();
			/* Move from other */
			plugins                  = std::move(other.plugins);
			nativePlugins            = std::move(other.nativePlugins);
			sourceRegistry           = std::move(other.sourceRegistry);
			taskRegistry             = std::move(other.taskRegistry);
			modelLoaderRegistry      = std::move(other.modelLoaderRegistry);
			transformRegistry        = std::move(other.transformRegistry);
			schedulerVtableRegistry  = std::move(other.schedulerVtableRegistry);
			wamrRefOwned             = other.wamrRefOwned;
			other.wamrRefOwned       = false;
		}
		return *this;
	}

	/* =========================================================================
	 * Plugin loading
	 * ====================================================================== */

	std::expected<void, std::string>
	PluginManager::load(const std::filesystem::path& path, std::string_view config_json) {
		const bool isWasm = (path.extension() == ".wasm");

		if (isWasm) {
			auto p = std::make_unique<Plugin>();
			PluginRegistrationCtx ctx;
			ctx.manager = this;
			ctx.wasmPlugin = p.get();
			auto hostApi = make_host_api(ctx);

			if (auto r = wasm_loader_load(path, config_json, hostApi, *p); !r) {
				return std::unexpected(r.error());
			}
			/* Switch persistentApi.ctx from the (now-stale) PluginRegistrationCtx*
			 * to this manager so that post-init host callbacks work correctly. */
			p->persistentApi.ctx = this;

			plugins.push_back(std::move(p));
		} else {
			auto p = std::make_unique<NativePlugin>();
			PluginRegistrationCtx ctx;
			ctx.manager = this;
			ctx.nativePlugin = p.get();
			auto hostApi = make_host_api(ctx);

			if (auto r = native_loader_load(path, config_json, hostApi, *p); !r) {
				return std::unexpected(r.error());
			}
			/* Switch persistentApi.ctx from the (now-stale) PluginRegistrationCtx*
			 * to this manager so that post-init host callbacks work correctly. */
			p->persistentApi.ctx = this;

			nativePlugins.push_back(std::move(p));
		}

		return {};
	}

	/* =========================================================================
	 * Source and task registries
	 * ====================================================================== */

	IDatasetSource* PluginManager::find_source(std::string_view scheme) const {
		const auto it = sourceRegistry.find(std::string(scheme));
		return (it != sourceRegistry.end()) ? it->second : nullptr;
	}

	ITask* PluginManager::find_task(std::string_view name_or_alias) const {
		const auto it = taskRegistry.find(std::string(name_or_alias));
		return (it != taskRegistry.end()) ? it->second : nullptr;
	}

	IModelLoader* PluginManager::find_model_loader(std::string_view path) const {
		for (auto* loader : modelLoaderRegistry) {
			if (loader->probe(path)) return loader;
		}
		return nullptr;
	}

	ITransform* PluginManager::find_transform(std::string_view name) const {
		const auto it = transformRegistry.find(std::string(name));
		return (it != transformRegistry.end()) ? it->second : nullptr;
	}

	void PluginManager::emit_model_loaded(std::string_view info_json) {
		for (auto& p : nativePlugins) {
			if (p->fnOnModelLoaded != nullptr) {
				p->fnOnModelLoaded(info_json.data(), static_cast<uint32_t>(info_json.size()));
			}
		}
		// WASM plugins: would need wasm_push_string like other hooks — future work
	}

	/* =========================================================================
	 * Host API construction
	 * ====================================================================== */

	ttm_host_api PluginManager::make_host_api(PluginRegistrationCtx& ctx) {
		/* Populate the C++ registration callbacks (called by WASM host callbacks) */
		ctx.attach_source = [this, &ctx](std::unique_ptr<IDatasetSource> src) {
			register_source_impl(std::move(src), ctx);
		};
		ctx.attach_task = [this, &ctx](std::unique_ptr<ITask> task) {
			register_task_impl(std::move(task), ctx);
		};
		ctx.attach_model_loader = [this, &ctx](std::unique_ptr<IModelLoader> loader) {
			register_model_loader_impl(std::move(loader), ctx);
		};
		ctx.attach_transform = [this, &ctx](std::unique_ptr<ITransform> transform) {
			register_transform_impl(std::move(transform), ctx);
		};

		ttm_host_api api{};
		api.ctx                  = &ctx;
		api.register_source      = &PluginManager::s_register_source;
		api.register_transform   = &PluginManager::s_register_transform;
		api.register_task        = &PluginManager::s_register_task;
		api.register_metric      = &PluginManager::s_register_metric;
		api.register_model_loader = &PluginManager::s_register_model_loader;
		api.notify_model_info    = &PluginManager::s_notify_model_info;
		api.register_scheduler   = &PluginManager::s_register_scheduler;
		api.log                  = &PluginManager::s_log;
		api.log_metric           = &PluginManager::s_log_metric;
		api.terminal_size        = &PluginManager::s_terminal_size;
		api.alloc                = &PluginManager::s_alloc;
		api.free                 = &PluginManager::s_free;
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

		auto* regCtx = static_cast<PluginRegistrationCtx*>(ctx);
		auto adapter = std::make_unique<CSourceAdapter>(std::move(schemeList), vt);
		regCtx->manager->register_source_impl(std::move(adapter), *regCtx);
		return TTM_OK;
	}

	void PluginManager::register_source_impl(std::unique_ptr<IDatasetSource> src, PluginRegistrationCtx& ctx) {
		for (const auto& scheme : src->schemes()) {
			if (sourceRegistry.contains(scheme)) {
				std::cerr << "[ttm] warning: source scheme '" << scheme << "' already registered; overriding.\n";
			}
			sourceRegistry[scheme] = src.get();
		}
		/* Attach ownership to the owning plugin record */
		if (ctx.nativePlugin != nullptr) {
			ctx.nativePlugin->sources.push_back(std::move(src));
		} else if (ctx.wasmPlugin != nullptr) {
			ctx.wasmPlugin->sources.push_back(std::move(src));
		} else {
			assert(!plugins.empty());
			plugins.back()->sources.push_back(std::move(src));
		}
	}

	ttm_error PluginManager::s_register_task(
			void* ctx, const char* name, const char** /*aliases_unused*/, const ttm_task_vtable* vt
	) {
		if (ctx == nullptr || name == nullptr || vt == nullptr) {
			return TTM_ERR_ARGS;
		}
		/* name and aliases come from the vtable itself; aliases_unused param is ignored */
		auto* regCtx = static_cast<PluginRegistrationCtx*>(ctx);
		auto adapter = std::make_unique<CTaskAdapter>(*vt);
		regCtx->manager->register_task_impl(std::move(adapter), *regCtx);
		return TTM_OK;
	}

	void PluginManager::register_task_impl(std::unique_ptr<ITask> task, PluginRegistrationCtx& ctx) {
		auto* raw = task.get();

		/* Register canonical name */
		const auto taskName = std::string(task->name());
		if (taskRegistry.contains(taskName)) {
			std::cerr << "[ttm] warning: task '" << taskName << "' already registered; overriding.\n";
		}
		taskRegistry[taskName] = raw;

		/* Register all aliases */
		for (auto alias : task->aliases()) {
			const auto aliasStr = std::string(alias);
			taskRegistry[aliasStr] = raw;
		}

		/* Transfer ownership to the owning plugin record */
		if (ctx.nativePlugin != nullptr) {
			ctx.nativePlugin->tasks.push_back(std::move(task));
		} else if (ctx.wasmPlugin != nullptr) {
			ctx.wasmPlugin->tasks.push_back(std::move(task));
		} else {
			assert(!plugins.empty());
			plugins.back()->tasks.push_back(std::move(task));
		}
	}

	ttm_error PluginManager::s_register_transform(
			void* ctx, const char* name, const char** /*aliases*/, const ttm_transform_vtable* vt
	) {
		if (ctx == nullptr || name == nullptr || vt == nullptr) return TTM_ERR_ARGS;
		auto* regCtx = static_cast<PluginRegistrationCtx*>(ctx);

		// Build a minimal ITransform adapter
		class CVtableTransform final : public ITransform {
		public:
			CVtableTransform(std::string n, const ttm_transform_vtable& v)
				: name_(std::move(n)), vt_(v), handle_(TTM_INVALID_HANDLE)
			{
				if (vt_.create != nullptr) {
					handle_ = vt_.create("{}", 2);
				}
			}
			~CVtableTransform() override {
				if (handle_ != TTM_INVALID_HANDLE && vt_.destroy != nullptr) {
					vt_.destroy(handle_);
				}
			}
			[[nodiscard]] std::string_view name() const override { return name_; }
			ttm_error apply_ipc(const void* in_ipc, uint32_t in_len,
			                    void** out_ipc, uint32_t* out_len) override {
				if (vt_.apply == nullptr || handle_ == TTM_INVALID_HANDLE) return TTM_ERR_UNSUPPORTED;
				return vt_.apply(handle_, in_ipc, in_len, out_ipc, out_len);
			}
		private:
			std::string             name_;
			ttm_transform_vtable    vt_;
			ttm_handle              handle_;
		};

		auto adapter = std::make_unique<CVtableTransform>(name, *vt);
		regCtx->manager->register_transform_impl(std::move(adapter), *regCtx);
		return TTM_OK;
	}

	void PluginManager::register_transform_impl(std::unique_ptr<ITransform> transform, PluginRegistrationCtx& ctx) {
		const auto key = std::string(transform->name());
		if (transformRegistry.contains(key)) {
			std::cerr << "[ttm] warning: transform '" << key << "' already registered; overriding.\n";
		}
		transformRegistry[key] = transform.get();
		if (ctx.nativePlugin != nullptr) {
			ctx.nativePlugin->transforms.push_back(std::move(transform));
		} else if (ctx.wasmPlugin != nullptr) {
			ctx.wasmPlugin->transforms.push_back(std::move(transform));
		} else {
			assert(!plugins.empty());
			plugins.back()->transforms.push_back(std::move(transform));
		}
	}

	ttm_error PluginManager::s_register_model_loader(void* ctx, const ttm_model_loader_vtable* vt) {
		if (ctx == nullptr || vt == nullptr) return TTM_ERR_ARGS;
		auto* regCtx = static_cast<PluginRegistrationCtx*>(ctx);
		auto adapter = std::make_unique<CModelLoaderAdapter>(vt);
		regCtx->manager->register_model_loader_impl(std::move(adapter), *regCtx);
		return TTM_OK;
	}

	void PluginManager::register_model_loader_impl(std::unique_ptr<IModelLoader> loader, PluginRegistrationCtx& ctx) {
		modelLoaderRegistry.push_back(loader.get());
		if (ctx.nativePlugin != nullptr) {
			ctx.nativePlugin->modelLoaders.push_back(std::move(loader));
		} else if (ctx.wasmPlugin != nullptr) {
			// WASM model loaders not yet supported; store in built-in plugin slot
			assert(!plugins.empty());
			plugins.back()->modelLoaders.push_back(std::move(loader));
		} else {
			assert(!plugins.empty());
			plugins.back()->modelLoaders.push_back(std::move(loader));
		}
	}

	void PluginManager::s_notify_model_info(void* ctx, const ttm_model_info_t* info) {
		if (ctx == nullptr || info == nullptr) return;
		// Serialise to JSON and broadcast via emit_model_loaded
		char buf[512];
		std::snprintf(buf, sizeof(buf),
			R"({"name":"%s","arch":"%s","num_parameters":%llu,"num_trainable":%llu,"bytes_on_device":%llu,"device_type":%d,"device_id":%d})",
			info->name ? info->name : "",
			info->arch ? info->arch : "",
			static_cast<unsigned long long>(info->num_parameters),
			static_cast<unsigned long long>(info->num_trainable),
			static_cast<unsigned long long>(info->bytes_on_device),
			static_cast<int>(info->device_type),
			static_cast<int>(info->device_id)
		);
		static_cast<PluginManager*>(ctx)->emit_model_loaded(std::string_view{buf});
	}

	ttm_error PluginManager::s_register_scheduler(void* ctx, const char* name, const ttm_scheduler_vtable* vt) {
		if (ctx == nullptr || name == nullptr || vt == nullptr) return TTM_ERR_ARGS;
		auto* regCtx = static_cast<PluginRegistrationCtx*>(ctx);
		regCtx->manager->schedulerVtableRegistry.insert_or_assign(std::string(name), *vt);
		return TTM_OK;
	}

	const ttm_scheduler_vtable* PluginManager::find_scheduler_vtable(std::string_view name) const {
		const auto it = schedulerVtableRegistry.find(std::string(name));
		return (it != schedulerVtableRegistry.end()) ? &it->second : nullptr;
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
	 * Each emit_* function iterates both WASM and native plugins and calls the
	 * resolved hook on plugins that exported it.  WASM plugins receive string
	 * arguments via WASM linear memory; native plugins receive raw C string ptrs.
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
		std::array<uint32_t, 2> args{wasmPtr, wasmLen};
		wasm_runtime_call_wasm(plug.env, fn, args.size(), args.data());
		wasm_runtime_module_free(plug.inst, wasmPtr);
	}

	void PluginManager::emit_fit_begin(std::string_view ctx_json) {
		for (auto& p : plugins) {
			call_with_string(*p, p->fnFitBegin, ctx_json);
		}
		for (auto& p : nativePlugins) {
			if (p->fnFitBegin != nullptr) {
				p->fnFitBegin(ctx_json.data(), static_cast<uint32_t>(ctx_json.size()));
			}
		}
	}

	void PluginManager::emit_epoch_begin(std::uint32_t epoch, std::uint32_t total) {
		for (auto& p : plugins) {
			if (p->fnEpochBegin == nullptr) {
				continue;
			}
			std::array<uint32_t, 2> args{epoch, total};
			wasm_runtime_call_wasm(p->env, p->fnEpochBegin, args.size(), args.data());
		}
		for (auto& p : nativePlugins) {
			if (p->fnEpochBegin != nullptr) {
				p->fnEpochBegin(epoch, total);
			}
		}
	}

	void PluginManager::emit_batch_begin(std::uint32_t batch, std::uint32_t total) {
		for (auto& p : plugins) {
			if (p->fnBatchBegin == nullptr) {
				continue;
			}
			std::array<uint32_t, 2> args{batch, total};
			wasm_runtime_call_wasm(p->env, p->fnBatchBegin, args.size(), args.data());
		}
		for (auto& p : nativePlugins) {
			if (p->fnBatchBegin != nullptr) {
				p->fnBatchBegin(batch, total);
			}
		}
	}

	float PluginManager::emit_loss_computed(float loss) {
		for (auto& p : plugins) {
			if (p->fnLossComputed == nullptr) {
				continue;
			}
			/* WAMR passes f32 as uint32 bit-cast */
			std::array<uint32_t, 1> args{};
			std::memcpy(&args[0], &loss, sizeof(float));
			wasm_runtime_call_wasm(p->env, p->fnLossComputed, args.size(), args.data());
			std::memcpy(&loss, &args[0], sizeof(float));
		}
		for (auto& p : nativePlugins) {
			if (p->fnLossComputed != nullptr) {
				loss = p->fnLossComputed(loss);
			}
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
			std::array<uint32_t, 4> args{batch, lossBits, metricsPtr, metricsLen};
			wasm_runtime_call_wasm(p->env, p->fnBatchEnd, args.size(), args.data());
			wasm_runtime_module_free(p->inst, metricsPtr);
		}
		for (auto& p : nativePlugins) {
			if (p->fnBatchEnd != nullptr) {
				p->fnBatchEnd(batch, loss, metrics_json.data(), static_cast<uint32_t>(metrics_json.size()));
			}
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
			std::array<uint32_t, 3> args{epoch, metricsPtr, metricsLen};
			wasm_runtime_call_wasm(p->env, p->fnEpochEnd, args.size(), args.data());
			wasm_runtime_module_free(p->inst, metricsPtr);
			if (args[0] != 0) {
				stop = true;
			}
		}
		for (auto& p : nativePlugins) {
			if (p->fnEpochEnd != nullptr) {
				if (p->fnEpochEnd(epoch, metrics_json.data(), static_cast<uint32_t>(metrics_json.size())) != 0) {
					stop = true;
				}
			}
		}
		return stop;
	}

	void PluginManager::emit_validation_end(std::string_view metrics_json) {
		for (auto& p : plugins) {
			call_with_string(*p, p->fnValidationEnd, metrics_json);
		}
		for (auto& p : nativePlugins) {
			if (p->fnValidationEnd != nullptr) {
				p->fnValidationEnd(metrics_json.data(), static_cast<uint32_t>(metrics_json.size()));
			}
		}
	}

	void PluginManager::emit_fit_end(std::string_view metrics_json) {
		for (auto& p : plugins) {
			call_with_string(*p, p->fnFitEnd, metrics_json);
		}
		for (auto& p : nativePlugins) {
			if (p->fnFitEnd != nullptr) {
				p->fnFitEnd(metrics_json.data(), static_cast<uint32_t>(metrics_json.size()));
			}
		}
	}

	void PluginManager::emit_log(ttm_log_level level, std::string_view msg) {
		bool handled = false;

		/* Dispatch to WASM plugins */
		for (auto& p : plugins) {
			if (p->fnOnLog == nullptr) continue;
			uint32_t wasmPtr = 0, wasmLen = 0;
			if (!wasm_push_string(p->inst, msg, wasmPtr, wasmLen)) continue;
			std::array<uint32_t, 3> args{static_cast<uint32_t>(level), wasmPtr, wasmLen};
			wasm_runtime_call_wasm(p->env, p->fnOnLog, args.size(), args.data());
			wasm_runtime_module_free(p->inst, wasmPtr);
			handled = true;
		}

		/* Dispatch to native plugins */
		for (auto& p : nativePlugins) {
			if (p->fnOnLog == nullptr) continue;
			p->fnOnLog(static_cast<uint32_t>(level), msg.data(), static_cast<uint32_t>(msg.size()));
			handled = true;
		}

		/* Fallback to stderr if no plugin consumed the message */
		if (!handled) {
			PluginManager::s_log(nullptr, level, msg.data(), static_cast<uint32_t>(msg.size()));
		}
	}

	void PluginManager::emit_metric(std::string_view key, float value, int32_t step) {
		/* Dispatch to WASM plugins */
		for (auto& p : plugins) {
			if (p->fnOnMetric == nullptr) continue;
			uint32_t keyPtr = 0, keyLen = 0;
			if (!wasm_push_string(p->inst, key, keyPtr, keyLen)) continue;
			uint32_t valueBits = 0;
			std::memcpy(&valueBits, &value, sizeof(float));
			std::array<uint32_t, 4> args{keyPtr, keyLen, valueBits, static_cast<uint32_t>(step)};
			wasm_runtime_call_wasm(p->env, p->fnOnMetric, args.size(), args.data());
			wasm_runtime_module_free(p->inst, keyPtr);
		}

		/* Dispatch to native plugins */
		for (auto& p : nativePlugins) {
			if (p->fnOnMetric == nullptr) continue;
			p->fnOnMetric(key.data(), static_cast<uint32_t>(key.size()), value, step);
		}
	}

	void PluginManager::s_log_metric(void* ctx, const char* key, uint32_t key_len, float value, int32_t step) {
		if (ctx == nullptr || key == nullptr || key_len == 0) return;
		static_cast<PluginManager*>(ctx)->emit_metric({key, key_len}, value, step);
	}

	void PluginManager::s_terminal_size(void* /*ctx*/, uint32_t* out_width, uint32_t* out_height) {
		uint32_t width = 80, height = 24;
#if defined(TIOCGWINSZ)
		struct winsize ws{};
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg) -- standard POSIX terminal size query
		if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
			if (ws.ws_col > 0) width  = static_cast<uint32_t>(ws.ws_col);
			if (ws.ws_row > 0) height = static_cast<uint32_t>(ws.ws_row);
		}
#endif
		if (out_width  != nullptr) *out_width  = width;
		if (out_height != nullptr) *out_height = height;
	}

} // namespace ttm::plugins

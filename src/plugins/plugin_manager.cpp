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
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

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
     * @throws std::runtime_error if the file cannot be opened.
     */
    explicit FileReader(const std::string& path)
        : file_(path, std::ios::binary)
    {
        if (!file_) {
            throw std::runtime_error("FileReader: cannot open '" + path + "'");
        }
    }

    std::streamsize read(std::byte* buf, std::streamsize n) override
    {
        file_.read(reinterpret_cast<char*>(buf), n);
        return file_.gcount();
    }

    bool seekable() const noexcept override { return true; }

    std::streampos seek(std::streamoff off, std::ios_base::seekdir dir) override
    {
        file_.seekg(off, dir);
        if (!file_) return std::streampos(-1);
        return file_.tellg();
    }

private:
    std::ifstream file_;
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
    std::vector<std::string> schemes() const override
    {
        return {"file:"};
    }

    std::unique_ptr<IByteReader> open(std::string_view uri) override
    {
        /* Strip "file://" or "file:" prefix */
        std::string path{uri};
        if (path.substr(0, 7) == "file://") {
            path = path.substr(7);
        } else if (path.substr(0, 5) == "file:") {
            path = path.substr(5);
        }

        try {
            return std::make_unique<FileReader>(path);
        } catch (const std::exception& ex) {
            std::cerr << "[ttm] FileSource::open error: " << ex.what() << '\n';
            return nullptr;
        }
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
    CVtableReader(const ttm_source_vtable& vt, ttm_handle handle)
        : vt_(vt), handle_(handle) {}

    ~CVtableReader() override
    {
        if (handle_ != TTM_INVALID_HANDLE && vt_.close) {
            vt_.close(handle_);
        }
    }

    std::streamsize read(std::byte* buf, std::streamsize n) override
    {
        if (!vt_.read) return -1;
        return static_cast<std::streamsize>(
            vt_.read(handle_, buf, static_cast<int32_t>(n)));
    }

    bool seekable() const noexcept override { return vt_.seek != nullptr; }

    std::streampos seek(std::streamoff off, std::ios_base::seekdir dir) override
    {
        if (!vt_.seek) return std::streampos(-1);
        int32_t whence = 0;
        switch (dir) {
            case std::ios_base::beg: whence = 0; break;
            case std::ios_base::cur: whence = 1; break;
            case std::ios_base::end: whence = 2; break;
            default: return std::streampos(-1);
        }
        const auto pos = vt_.seek(handle_, static_cast<int64_t>(off), whence);
        return (pos < 0) ? std::streampos(-1) : std::streampos(pos);
    }

private:
    const ttm_source_vtable& vt_;
    ttm_handle                handle_;
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
     * @param[in] scheme_list  Schemes this source handles (copied).
     * @param[in] vt           Source vtable (must remain valid for this object's lifetime).
     */
    CSourceAdapter(std::vector<std::string> scheme_list,
                   const ttm_source_vtable* vt)
        : schemes_(std::move(scheme_list)), vt_(*vt) {}

    std::vector<std::string> schemes() const override { return schemes_; }

    std::unique_ptr<IByteReader> open(std::string_view uri) override
    {
        char err_buf[256] = {};
        const auto handle = vt_.open(
            uri.data(), static_cast<uint32_t>(uri.size()),
            err_buf, sizeof(err_buf));

        if (handle == TTM_INVALID_HANDLE) {
            std::cerr << "[ttm] CSourceAdapter::open error: " << err_buf << '\n';
            return nullptr;
        }
        return std::make_unique<CVtableReader>(vt_, handle);
    }

private:
    std::vector<std::string> schemes_;
    ttm_source_vtable        vt_;  /* Copy of the vtable struct */
};

} // anonymous namespace

/* =========================================================================
 * PluginManager — construction / destruction
 * ====================================================================== */

PluginManager::PluginManager()
{
    wasm_loader_init();

    /* Register the built-in local filesystem source */
    auto file_src = std::make_unique<FileSource>();
    for (const auto& scheme : file_src->schemes()) {
        source_registry_.emplace(scheme, file_src.get());
    }
    /* Store a dummy Plugin record to own the built-in source */
    auto builtin_plugin = std::make_unique<Plugin>();
    builtin_plugin->sources.push_back(std::move(file_src));
    plugins_.push_back(std::move(builtin_plugin));
}

PluginManager::~PluginManager()
{
    /* Unload all plugins (index 0 is the built-in, no WASM handles to release) */
    for (auto& p : plugins_) {
        wasm_loader_unload(*p);
    }
    plugins_.clear();
    source_registry_.clear();

    wasm_loader_destroy();
}

/* =========================================================================
 * Plugin loading
 * ====================================================================== */

void PluginManager::load(const std::filesystem::path& path,
                          std::string_view config_json)
{
    auto p        = std::make_unique<Plugin>();
    auto host_api = make_host_api();

    wasm_loader_load(path, config_json, host_api, *p);

    plugins_.push_back(std::move(p));
}

/* =========================================================================
 * Source registry
 * ====================================================================== */

IDatasetSource* PluginManager::find_source(std::string_view scheme) const
{
    const auto it = source_registry_.find(std::string(scheme));
    return (it != source_registry_.end()) ? it->second : nullptr;
}

/* =========================================================================
 * Host API construction
 * ====================================================================== */

ttm_host_api PluginManager::make_host_api()
{
    ttm_host_api api{};
    api.ctx                = this;
    api.register_source    = &PluginManager::s_register_source;
    api.register_transform = &PluginManager::s_register_transform;
    api.register_task      = &PluginManager::s_register_task;
    api.register_metric    = &PluginManager::s_register_metric;
    api.log                = &PluginManager::s_log;
    api.alloc              = &PluginManager::s_alloc;
    api.free               = &PluginManager::s_free;
    return api;
}

/* =========================================================================
 * Host API static callbacks
 * ====================================================================== */

ttm_error PluginManager::s_register_source(void* ctx,
                                            const char** schemes,
                                            const ttm_source_vtable* vt)
{
    if (!ctx || !schemes || !vt) return TTM_ERR_ARGS;

    std::vector<std::string> scheme_list;
    for (const char** s = schemes; *s != nullptr; ++s) {
        scheme_list.emplace_back(*s);
    }
    if (scheme_list.empty()) return TTM_ERR_ARGS;

    auto* self = static_cast<PluginManager*>(ctx);
    auto  adapter = std::make_unique<CSourceAdapter>(std::move(scheme_list), vt);

    /* The plugin currently being loaded is always the last one pushed before
     * the init call; we pass nullptr here and let register_source_impl
     * attach the source to the most-recently-added plugin. */
    self->register_source_impl(std::move(adapter), nullptr);
    return TTM_OK;
}

void PluginManager::register_source_impl(std::unique_ptr<IDatasetSource> src,
                                          Plugin* /*owner — reserved for future use*/)
{
    for (const auto& scheme : src->schemes()) {
        if (source_registry_.count(scheme)) {
            std::cerr << "[ttm] warning: source scheme '" << scheme
                      << "' already registered; overriding.\n";
        }
        source_registry_[scheme] = src.get();
    }
    /* Attach to the last plugin (the one currently being initialised) */
    assert(!plugins_.empty());
    plugins_.back()->sources.push_back(std::move(src));
}

ttm_error PluginManager::s_register_transform(void* /*ctx*/,
                                               const char* /*name*/,
                                               const char** /*aliases*/,
                                               const ttm_transform_vtable* /*vt*/)
{
    /* TODO: implement transform registry */
    return TTM_ERR_UNSUPPORTED;
}

ttm_error PluginManager::s_register_task(void* /*ctx*/,
                                          const char* /*name*/,
                                          const char** /*aliases*/,
                                          const ttm_task_vtable* /*vt*/)
{
    /* TODO: implement task registry */
    return TTM_ERR_UNSUPPORTED;
}

ttm_error PluginManager::s_register_metric(void* /*ctx*/,
                                            const char* /*name*/,
                                            const char** /*aliases*/,
                                            const ttm_metric_vtable* /*vt*/)
{
    /* TODO: implement metric registry */
    return TTM_ERR_UNSUPPORTED;
}

void PluginManager::s_log(void* /*ctx*/, ttm_log_level level,
                           const char* msg, uint32_t len)
{
    const char* prefix = "";
    switch (level) {
        case TTM_LOG_TRACE: prefix = "[TRACE] "; break;
        case TTM_LOG_DEBUG: prefix = "[DEBUG] "; break;
        case TTM_LOG_INFO:  prefix = "[INFO]  "; break;
        case TTM_LOG_WARN:  prefix = "[WARN]  "; break;
        case TTM_LOG_ERROR: prefix = "[ERROR] "; break;
    }
    std::cerr << "[ttm plugin] " << prefix;
    std::cerr.write(msg, static_cast<std::streamsize>(len));
    std::cerr << '\n';
}

void* PluginManager::s_alloc(void* /*ctx*/, uint32_t size)
{
    return std::malloc(size);
}

void PluginManager::s_free(void* /*ctx*/, void* ptr)
{
    std::free(ptr);
}

/* =========================================================================
 * Lifecycle event dispatch
 *
 * Each emit_* function iterates plugins_ and calls the resolved hook on
 * plugins that exported it.  String arguments are pushed into WASM linear
 * memory for each call and freed immediately after.
 * ====================================================================== */

/** @brief Helper — push a string to WASM memory, call fn, then free. */
static void call_with_string(Plugin& p,
                              wasm_function_inst_t fn,
                              std::string_view str)
{
    if (!fn) return;
    uint32_t wasm_ptr = 0, wasm_len = 0;
    wasm_push_string(p.inst, str, wasm_ptr, wasm_len);
    uint32_t args[2] = { wasm_ptr, wasm_len };
    wasm_runtime_call_wasm(p.env, fn, 2, args);
    wasm_runtime_module_free(p.inst, wasm_ptr);
}

void PluginManager::emit_fit_begin(std::string_view ctx_json)
{
    for (auto& p : plugins_) {
        call_with_string(*p, p->fn_fit_begin, ctx_json);
    }
}

void PluginManager::emit_epoch_begin(std::uint32_t epoch, std::uint32_t total)
{
    for (auto& p : plugins_) {
        if (!p->fn_epoch_begin) continue;
        uint32_t args[2] = { epoch, total };
        wasm_runtime_call_wasm(p->env, p->fn_epoch_begin, 2, args);
    }
}

void PluginManager::emit_batch_begin(std::uint32_t batch, std::uint32_t total)
{
    for (auto& p : plugins_) {
        if (!p->fn_batch_begin) continue;
        uint32_t args[2] = { batch, total };
        wasm_runtime_call_wasm(p->env, p->fn_batch_begin, 2, args);
    }
}

float PluginManager::emit_loss_computed(float loss)
{
    for (auto& p : plugins_) {
        if (!p->fn_loss_computed) continue;
        /* WAMR passes f32 as uint32 bit-cast */
        uint32_t args[1];
        std::memcpy(&args[0], &loss, sizeof(float));
        wasm_runtime_call_wasm(p->env, p->fn_loss_computed, 1, args);
        std::memcpy(&loss, &args[0], sizeof(float));
    }
    return loss;
}

void PluginManager::emit_batch_end(std::uint32_t batch, float loss,
                                    std::string_view metrics_json)
{
    for (auto& p : plugins_) {
        if (!p->fn_batch_end) continue;
        uint32_t metrics_ptr = 0, metrics_len = 0;
        wasm_push_string(p->inst, metrics_json, metrics_ptr, metrics_len);
        uint32_t loss_bits;
        std::memcpy(&loss_bits, &loss, sizeof(float));
        uint32_t args[4] = { batch, loss_bits, metrics_ptr, metrics_len };
        wasm_runtime_call_wasm(p->env, p->fn_batch_end, 4, args);
        wasm_runtime_module_free(p->inst, metrics_ptr);
    }
}

bool PluginManager::emit_epoch_end(std::uint32_t epoch,
                                    std::string_view metrics_json)
{
    bool stop = false;
    for (auto& p : plugins_) {
        if (!p->fn_epoch_end) continue;
        uint32_t metrics_ptr = 0, metrics_len = 0;
        wasm_push_string(p->inst, metrics_json, metrics_ptr, metrics_len);
        uint32_t args[3] = { epoch, metrics_ptr, metrics_len };
        wasm_runtime_call_wasm(p->env, p->fn_epoch_end, 3, args);
        wasm_runtime_module_free(p->inst, metrics_ptr);
        if (args[0] != 0) stop = true;
    }
    return stop;
}

void PluginManager::emit_validation_end(std::string_view metrics_json)
{
    for (auto& p : plugins_) {
        call_with_string(*p, p->fn_validation_end, metrics_json);
    }
}

void PluginManager::emit_fit_end(std::string_view metrics_json)
{
    for (auto& p : plugins_) {
        call_with_string(*p, p->fn_fit_end, metrics_json);
    }
}

} // namespace ttm::plugins

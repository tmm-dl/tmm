/**
 * @file wasm_loader.cpp
 * @brief WAMR-based plugin loader implementation.
 *
 * @details
 * All WAMR API usage is confined to this translation unit so that the rest
 * of the plugin library does not pull in WAMR headers.
 */

#include "wasm_loader.hpp"

#include <ttm/plugins/abi.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <wasm_export.h>

namespace ttm::plugins {

/* =========================================================================
 * Runtime init / destroy — refcounted so multiple PluginManager instances
 * (e.g. in unit tests) are safe.
 * ====================================================================== */

namespace {
std::atomic<int> g_wamr_refcount{0};
} // namespace

void wasm_loader_init()
{
    if (g_wamr_refcount.fetch_add(1, std::memory_order_acq_rel) == 0) {
        RuntimeInitArgs args{};
        args.mem_alloc_type = Alloc_With_System_Allocator;

        if (!wasm_runtime_full_init(&args)) {
            g_wamr_refcount.fetch_sub(1, std::memory_order_acq_rel);
            throw std::runtime_error("wasm_loader_init: wasm_runtime_full_init failed");
        }
    }
}

void wasm_loader_destroy()
{
    if (g_wamr_refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        wasm_runtime_destroy();
    }
}

/* =========================================================================
 * String helpers
 * ====================================================================== */

void wasm_push_string(wasm_module_inst_t inst,
                      std::string_view str,
                      uint32_t& out_wasm_ptr,
                      uint32_t& out_len)
{
    const auto len = static_cast<uint32_t>(str.size());
    void* native_ptr = nullptr;
    const uint32_t wasm_ptr = wasm_runtime_module_malloc(inst, len, &native_ptr);
    if (wasm_ptr == 0) {
        throw std::runtime_error(
            "wasm_push_string: failed to allocate " + std::to_string(len)
            + " bytes in WASM linear memory");
    }
    std::memcpy(native_ptr, str.data(), len);
    out_wasm_ptr = wasm_ptr;
    out_len      = len;
}

/* =========================================================================
 * Host import registrations
 *
 * These are the functions exported by the host under the "ttm" module
 * namespace.  They are registered before the WASM module is instantiated so
 * that the linker can resolve the plugin's imports.
 *
 * Signature strings use WAMR's compact notation:
 *   i = i32, I = i64, f = f32, F = f64, * = pointer (i32 in wasm32)
 * ====================================================================== */

namespace {

/* ---------- log --------------------------------------------------------- */
static void host_log(wasm_exec_env_t env,
                     uint32_t level,
                     uint32_t msg_ptr,
                     uint32_t msg_len)
{
    auto* inst = wasm_runtime_get_module_inst(env);
    const auto* msg = static_cast<const char*>(
        wasm_runtime_addr_app_to_native(inst, msg_ptr));
    if (!msg) return;

    auto* api = static_cast<const ttm_host_api*>(
        wasm_runtime_get_user_data(env));
    if (api && api->log) {
        api->log(api->ctx, static_cast<ttm_log_level>(level), msg, msg_len);
    }
}

/* ---------- alloc ------------------------------------------------------- */
static uint32_t host_alloc(wasm_exec_env_t env, uint32_t size)
{
    auto* inst = wasm_runtime_get_module_inst(env);
    void* native_ptr = nullptr;
    return wasm_runtime_module_malloc(inst, size, &native_ptr);
}

/* ---------- free -------------------------------------------------------- */
static void host_free(wasm_exec_env_t env, uint32_t wasm_ptr)
{
    auto* inst = wasm_runtime_get_module_inst(env);
    wasm_runtime_module_free(inst, wasm_ptr);
}

/* ---------- register_source -------------------------------------------- */
static int32_t host_register_source(wasm_exec_env_t /*env*/,
                                    uint32_t /*schemes_ptr*/,
                                    uint32_t /*vtable_ptr*/)
{
    /* TODO: unmarshal the scheme list and vtable from WASM linear memory,
     * construct a CSourceAdapter, and forward to PluginManager via the
     * ttm_host_api ctx.  Stubbed for the initial build. */
    return TTM_ERR_UNSUPPORTED;
}

/* ---------- NativeSymbol table ----------------------------------------- */
static NativeSymbol ttm_native_symbols[] = {
    /* { "export_name", func_ptr, "signature", attachment } */
    { "ttm_log",             reinterpret_cast<void*>(host_log),             "(iii)",  nullptr },
    { "ttm_alloc",           reinterpret_cast<void*>(host_alloc),           "(i)i",   nullptr },
    { "ttm_free",            reinterpret_cast<void*>(host_free),            "(i)",    nullptr },
    { "ttm_register_source", reinterpret_cast<void*>(host_register_source), "(ii)i",  nullptr },
};

} // anonymous namespace

/* =========================================================================
 * Load / unload
 * ====================================================================== */

void wasm_loader_load(const std::filesystem::path& path,
                      std::string_view config_json,
                      const ttm_host_api& host_api,
                      Plugin& plugin)
{
    /* 1. Read file bytes ------------------------------------------------- */
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("wasm_loader_load: cannot open '" + path.string() + "'");
    }
    const std::vector<uint8_t> bytes(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    /* 2. Register host symbols ------------------------------------------- */
    if (!wasm_runtime_register_natives(
            "ttm",
            ttm_native_symbols,
            sizeof(ttm_native_symbols) / sizeof(ttm_native_symbols[0])))
    {
        throw std::runtime_error("wasm_loader_load: failed to register host symbols");
    }

    /* 3. Load (compile) module ------------------------------------------- */
    char error_buf[256] = {};
    plugin.module = wasm_runtime_load(
        const_cast<uint8_t*>(bytes.data()),
        static_cast<uint32_t>(bytes.size()),
        error_buf, sizeof(error_buf));
    if (!plugin.module) {
        throw std::runtime_error(
            std::string("wasm_loader_load: wasm_runtime_load failed: ") + error_buf);
    }

    /* 4. Instantiate ------------------------------------------------------- */
    constexpr uint32_t STACK_SIZE = 512 * 1024;  /* 512 KB */
    constexpr uint32_t HEAP_SIZE  = 4 * 1024 * 1024; /* 4 MB */
    plugin.inst = wasm_runtime_instantiate(
        plugin.module, STACK_SIZE, HEAP_SIZE, error_buf, sizeof(error_buf));
    if (!plugin.inst) {
        wasm_runtime_unload(plugin.module);
        plugin.module = nullptr;
        throw std::runtime_error(
            std::string("wasm_loader_load: instantiate failed: ") + error_buf);
    }

    /* 5. Create execution environment ------------------------------------- */
    plugin.env = wasm_runtime_create_exec_env(plugin.inst, STACK_SIZE);
    if (!plugin.env) {
        wasm_runtime_deinstantiate(plugin.inst);
        wasm_runtime_unload(plugin.module);
        plugin.inst   = nullptr;
        plugin.module = nullptr;
        throw std::runtime_error("wasm_loader_load: failed to create exec env");
    }

    /* Attach host API pointer so host callbacks can retrieve it */
    wasm_runtime_set_user_data(plugin.env,
        const_cast<ttm_host_api*>(&host_api));

    /* 6. Verify ABI version ----------------------------------------------- */
    auto* fn_info = wasm_runtime_lookup_function(plugin.inst, "ttm_plugin_get_info");
    if (!fn_info) {
        throw std::runtime_error(
            "wasm_loader_load: '" + path.string()
            + "' does not export ttm_plugin_get_info");
    }
    uint32_t info_args[1] = {};
    if (!wasm_runtime_call_wasm(plugin.env, fn_info, 0, info_args)) {
        throw std::runtime_error(
            "wasm_loader_load: ttm_plugin_get_info call failed: "
            + std::string(wasm_runtime_get_exception(plugin.inst)));
    }
    /* info_args[0] is the WASM pointer to ttm_plugin_info */
    auto* info = static_cast<const ttm_plugin_info*>(
        wasm_runtime_addr_app_to_native(plugin.inst, info_args[0]));
    if (!info) {
        throw std::runtime_error("wasm_loader_load: ttm_plugin_get_info returned null");
    }
    if (info->abi_version != TTM_ABI_VERSION) {
        throw std::runtime_error(
            "wasm_loader_load: plugin ABI version mismatch (plugin="
            + std::to_string(info->abi_version)
            + ", host=" + std::to_string(TTM_ABI_VERSION) + ")");
    }

    /* 7. Call ttm_plugin_init -------------------------------------------- */
    auto* fn_init = wasm_runtime_lookup_function(plugin.inst, "ttm_plugin_init");
    if (!fn_init) {
        throw std::runtime_error(
            "wasm_loader_load: '" + path.string()
            + "' does not export ttm_plugin_init");
    }

    /* Push config JSON into WASM linear memory */
    uint32_t config_wasm_ptr = 0, config_len = 0;
    wasm_push_string(plugin.inst, config_json, config_wasm_ptr, config_len);

    /* Push host_api struct into WASM linear memory */
    void* api_native_ptr = nullptr;
    const uint32_t api_wasm_ptr = wasm_runtime_module_malloc(
        plugin.inst, sizeof(ttm_host_api), &api_native_ptr);
    if (!api_wasm_ptr) {
        throw std::runtime_error("wasm_loader_load: failed to allocate host_api in WASM heap");
    }
    std::memcpy(api_native_ptr, &host_api, sizeof(ttm_host_api));

    /* init(host_api_ptr, config_ptr, config_len) → i32 */
    uint32_t init_args[3] = { api_wasm_ptr, config_wasm_ptr, config_len };
    if (!wasm_runtime_call_wasm(plugin.env, fn_init, 3, init_args)) {
        throw std::runtime_error(
            "wasm_loader_load: ttm_plugin_init failed: "
            + std::string(wasm_runtime_get_exception(plugin.inst)));
    }
    const auto init_result = static_cast<ttm_error>(init_args[0]);
    if (init_result != TTM_OK) {
        throw std::runtime_error(
            "wasm_loader_load: ttm_plugin_init returned error "
            + std::to_string(static_cast<int>(init_result)));
    }

    /* Free temporaries in WASM heap */
    wasm_runtime_module_free(plugin.inst, config_wasm_ptr);
    wasm_runtime_module_free(plugin.inst, api_wasm_ptr);

    /* 8. Resolve optional lifecycle hooks -------------------------------- */
    auto lookup = [&](const char* name) -> wasm_function_inst_t {
        return wasm_runtime_lookup_function(plugin.inst, name);
    };
    plugin.fn_fit_begin      = lookup("ttm_on_fit_begin");
    plugin.fn_epoch_begin    = lookup("ttm_on_epoch_begin");
    plugin.fn_batch_begin    = lookup("ttm_on_batch_begin");
    plugin.fn_loss_computed  = lookup("ttm_on_loss_computed");
    plugin.fn_batch_end      = lookup("ttm_on_batch_end");
    plugin.fn_epoch_end      = lookup("ttm_on_epoch_end");
    plugin.fn_validation_end = lookup("ttm_on_validation_end");
    plugin.fn_fit_end        = lookup("ttm_on_fit_end");
    plugin.fn_teardown       = lookup("ttm_plugin_teardown");
}

void wasm_loader_unload(Plugin& plugin)
{
    if (plugin.env && plugin.fn_teardown) {
        wasm_runtime_call_wasm(plugin.env, plugin.fn_teardown, 0, nullptr);
    }
    if (plugin.env) {
        wasm_runtime_destroy_exec_env(plugin.env);
        plugin.env = nullptr;
    }
    if (plugin.inst) {
        wasm_runtime_deinstantiate(plugin.inst);
        plugin.inst = nullptr;
    }
    if (plugin.module) {
        wasm_runtime_unload(plugin.module);
        plugin.module = nullptr;
    }
}

} // namespace ttm::plugins

/**
 * @file python_optimizer.cpp
 * @brief TTM Python plugin — AdamW optimizer backed by torch.optim.AdamW.
 *
 * @details
 * Registered under the name "adamw".  Creates a torch.optim.AdamW wrapping the
 * parameters of an already-loaded torch.nn.Module, identified by its model handle.
 *
 * ### Device placement
 * When the optimizer is created, the model is moved to the requested device
 * (parsed from the "device" field in cfg_json, e.g. "cpu", "cuda", "cuda:1").
 * The AdamW optimizer follows the parameter tensors and therefore runs on the
 * same device automatically.
 *
 * ### LR scheduler integration
 * The host calls set_lr() after each scheduler step.  This updates the lr
 * field on every param group so that the next step() applies the new rate.
 *
 * ### Cross-plugin compatibility
 * create() accepts host-allocated param/grad DLTensors for future cross-plugin
 * use (e.g. TVM model + PyTorch optimizer).  When model_h is valid and refers
 * to a model in this plugin's slot table, model.parameters() is used directly
 * (the natural PyTorch approach).  Cross-plugin DLTensor wrapping via DLPack
 * is reserved for a future implementation.
 */

#include "python_plugin.hpp"

#include <ttm_python_export.h>

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

/* ============================================================================
 * Constants
 * ========================================================================= */

static constexpr int kMaxOpts = 16;

/* ============================================================================
 * Optimizer slot table
 * ========================================================================= */

namespace {

struct PyOptState {
	PyObject* optimizer = nullptr;
	bool      used      = false;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyOptState g_opts[kMaxOpts]{};

ttm_handle alloc_opt_slot(PyObject* opt) {
	for (int i = 0; i < kMaxOpts; ++i) {
		if (!g_opts[i].used) {
			g_opts[i] = {opt, true};
			return static_cast<ttm_handle>(i);
		}
	}
	return TTM_INVALID_HANDLE;
}

PyOptState* get_opt(ttm_handle h) {
	if (h < 0 || h >= kMaxOpts) return nullptr;
	return g_opts[static_cast<int>(h)].used ? &g_opts[static_cast<int>(h)] : nullptr;
}

/* ============================================================================
 * Minimal JSON scalar helpers (no external library)
 * ========================================================================= */

float jsonFloat(const char* json, uint32_t len, const char* key, float def) {
	if (json == nullptr || len == 0) return def;
	const std::string_view j{json, len};
	const std::string search = std::string("\"") + key + "\"";
	auto pos = j.find(search);
	if (pos == std::string_view::npos) return def;
	pos = j.find(':', pos + search.size());
	if (pos == std::string_view::npos) return def;
	++pos;
	while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) { ++pos; }
	float val = def;
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	std::sscanf(j.data() + pos, "%f", &val);
	return val;
}

int64_t jsonInt(const char* json, uint32_t len, const char* key, int64_t def) {
	if (json == nullptr || len == 0) return def;
	const std::string_view j{json, len};
	const std::string search = std::string("\"") + key + "\"";
	auto pos = j.find(search);
	if (pos == std::string_view::npos) return def;
	pos = j.find(':', pos + search.size());
	if (pos == std::string_view::npos) return def;
	++pos;
	while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) { ++pos; }
	long long val = static_cast<long long>(def);
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	std::sscanf(j.data() + pos, "%lld", &val);
	return static_cast<int64_t>(val);
}

/** @brief Extract a JSON string value.  Returns @p def if the key is absent. */
std::string jsonString(const char* json, uint32_t len, const char* key, const char* def) {
	if (json == nullptr || len == 0) return def;
	const std::string_view j{json, len};
	const std::string search = std::string("\"") + key + "\"";
	auto pos = j.find(search);
	if (pos == std::string_view::npos) return def;
	pos = j.find(':', pos + search.size());
	if (pos == std::string_view::npos) return def;
	++pos;
	while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) { ++pos; }
	if (pos >= j.size() || j[pos] != '"') return def;
	++pos; // skip opening quote
	const auto end = j.find('"', pos);
	if (end == std::string_view::npos) return def;
	return std::string(j.substr(pos, end - pos));
}

} // anonymous namespace

/* ============================================================================
 * AdamW vtable implementations
 * ========================================================================= */

static ttm_handle py_adamw_create(
		ttm_handle      model_h,
		const DLTensor* /*params*/,    uint32_t /*param_count*/,
		const DLTensor* /*grads*/,
		const char*     cfg,           uint32_t cfg_len,
		char*           err,           uint32_t err_cap
) {
#ifndef TTM_PYTHON_HAS_TORCH
	std::snprintf(err, err_cap,
		"AdamW optimizer requires PyTorch.\n"
		"Install PyTorch (pip install torch) and rebuild the plugin.");
	return TTM_INVALID_HANDLE;
#else
	PyObject* model_obj = py_get_model_obj(model_h);
	if (model_obj == nullptr) {
		std::snprintf(err, err_cap,
			"AdamW: invalid model handle %lld — model must be loaded first",
			static_cast<long long>(model_h));
		return TTM_INVALID_HANDLE;
	}

	/* Parse optimizer hyperparameters from cfg_json */
	const float       lr           = jsonFloat(cfg, cfg_len, "lr",           1e-3f);
	const float       weight_decay = jsonFloat(cfg, cfg_len, "weight_decay", 1e-2f);
	const float       beta1        = jsonFloat(cfg, cfg_len, "beta1",        0.9f);
	const float       beta2        = jsonFloat(cfg, cfg_len, "beta2",        0.999f);
	const float       eps          = jsonFloat(cfg, cfg_len, "eps",          1e-8f);
	const int64_t     amsgrad      = jsonInt  (cfg, cfg_len, "amsgrad",      0);
	const std::string device       = jsonString(cfg, cfg_len, "device",      "cpu");

	/* Move model to the requested device ----------------------------------- */
	PyObject* device_str = PyUnicode_FromString(device.c_str());
	PyObject* to_result  = PyObject_CallMethod(model_obj, "to", "O", device_str);
	Py_DECREF(device_str);
	if (to_result == nullptr) {
		PyErr_Print();
		PyErr_Clear();
		std::snprintf(err, err_cap,
			"AdamW: failed to move model to device '%s' — "
			"check that the device is available", device.c_str());
		return TTM_INVALID_HANDLE;
	}
	Py_DECREF(to_result);

	/* Collect model.parameters() ------------------------------------------ */
	PyObject* params_iter = PyObject_CallMethod(model_obj, "parameters", nullptr);
	if (params_iter == nullptr) {
		PyErr_Clear();
		std::snprintf(err, err_cap, "AdamW: model.parameters() failed");
		return TTM_INVALID_HANDLE;
	}

	/* Import torch.optim.AdamW -------------------------------------------- */
	PyObject* torch_mod   = PyImport_ImportModule("torch");
	PyObject* optim_mod   = torch_mod  ? PyObject_GetAttrString(torch_mod,  "optim") : nullptr;
	PyObject* adamw_class = optim_mod  ? PyObject_GetAttrString(optim_mod,  "AdamW") : nullptr;
	Py_XDECREF(torch_mod);
	Py_XDECREF(optim_mod);

	if (adamw_class == nullptr) {
		PyErr_Clear();
		Py_DECREF(params_iter);
		std::snprintf(err, err_cap, "AdamW: cannot access torch.optim.AdamW");
		return TTM_INVALID_HANDLE;
	}

	/* Build keyword-argument dict ----------------------------------------- */
	PyObject* kwargs = PyDict_New();

	PyObject* lr_obj = PyFloat_FromDouble(static_cast<double>(lr));
	PyDict_SetItemString(kwargs, "lr", lr_obj);
	Py_DECREF(lr_obj);

	PyObject* wd_obj = PyFloat_FromDouble(static_cast<double>(weight_decay));
	PyDict_SetItemString(kwargs, "weight_decay", wd_obj);
	Py_DECREF(wd_obj);

	PyObject* b1_obj = PyFloat_FromDouble(static_cast<double>(beta1));
	PyObject* b2_obj = PyFloat_FromDouble(static_cast<double>(beta2));
	PyObject* betas  = PyTuple_Pack(2, b1_obj, b2_obj);
	Py_DECREF(b1_obj);
	Py_DECREF(b2_obj);
	PyDict_SetItemString(kwargs, "betas", betas);
	Py_DECREF(betas);

	PyObject* eps_obj = PyFloat_FromDouble(static_cast<double>(eps));
	PyDict_SetItemString(kwargs, "eps", eps_obj);
	Py_DECREF(eps_obj);

	PyObject* amsgrad_obj = PyBool_FromLong(static_cast<long>(amsgrad));
	PyDict_SetItemString(kwargs, "amsgrad", amsgrad_obj);
	Py_DECREF(amsgrad_obj);

	/* Instantiate AdamW(model.parameters(), **kwargs) --------------------- */
	PyObject* pos_args = PyTuple_Pack(1, params_iter);
	Py_DECREF(params_iter);

	PyObject* optimizer = PyObject_Call(adamw_class, pos_args, kwargs);
	Py_DECREF(adamw_class);
	Py_DECREF(pos_args);
	Py_DECREF(kwargs);

	if (optimizer == nullptr) {
		PyErr_Print();
		PyErr_Clear();
		std::snprintf(err, err_cap, "AdamW: torch.optim.AdamW() constructor failed");
		return TTM_INVALID_HANDLE;
	}

	const ttm_handle h = alloc_opt_slot(optimizer);
	if (h == TTM_INVALID_HANDLE) {
		std::snprintf(err, err_cap, "AdamW: too many optimizers loaded simultaneously (max %d)", kMaxOpts);
		Py_DECREF(optimizer);
	}
	return h;
#endif // TTM_PYTHON_HAS_TORCH
}

static ttm_error py_adamw_step(ttm_handle h) {
	auto* st = get_opt(h);
	if (st == nullptr) return TTM_ERR_NOT_FOUND;
#ifdef TTM_PYTHON_HAS_TORCH
	PyObject* result = PyObject_CallMethod(st->optimizer, "step", nullptr);
	if (result == nullptr) { PyErr_Clear(); return TTM_ERR_IO; }
	Py_DECREF(result);
#endif
	return TTM_OK;
}

static ttm_error py_adamw_zero_grad(ttm_handle h) {
	auto* st = get_opt(h);
	if (st == nullptr) return TTM_ERR_NOT_FOUND;
#ifdef TTM_PYTHON_HAS_TORCH
	PyObject* result = PyObject_CallMethod(st->optimizer, "zero_grad", nullptr);
	if (result == nullptr) { PyErr_Clear(); return TTM_ERR_IO; }
	Py_DECREF(result);
#endif
	return TTM_OK;
}

static float py_adamw_get_lr(ttm_handle h) {
	const auto* st = get_opt(h);
	if (st == nullptr) return 0.0f;
#ifdef TTM_PYTHON_HAS_TORCH
	/* optimizer.param_groups[0]['lr'] */
	PyObject* groups = PyObject_GetAttrString(st->optimizer, "param_groups");
	if (groups == nullptr) { PyErr_Clear(); return 0.0f; }
	PyObject* first = PyList_Size(groups) > 0 ? PyList_GET_ITEM(groups, 0) : nullptr;
	if (first == nullptr) { Py_DECREF(groups); return 0.0f; }
	PyObject* lr_obj = PyDict_GetItemString(first, "lr"); // borrowed ref
	const float lr   = lr_obj ? static_cast<float>(PyFloat_AsDouble(lr_obj)) : 0.0f;
	Py_DECREF(groups);
	return lr;
#else
	return 0.0f;
#endif
}

static void py_adamw_set_lr(ttm_handle h, float lr) {
	auto* st = get_opt(h);
	if (st == nullptr) return;
#ifdef TTM_PYTHON_HAS_TORCH
	/* Update lr on every param group so all parameters use the new rate. */
	PyObject* groups = PyObject_GetAttrString(st->optimizer, "param_groups");
	if (groups == nullptr) { PyErr_Clear(); return; }
	PyObject* lr_obj = PyFloat_FromDouble(static_cast<double>(lr));
	const Py_ssize_t n = PyList_Size(groups);
	for (Py_ssize_t i = 0; i < n; ++i) {
		PyObject* pg = PyList_GET_ITEM(groups, i); // borrowed
		PyDict_SetItemString(pg, "lr", lr_obj);
	}
	Py_DECREF(lr_obj);
	Py_DECREF(groups);
#endif
}

static void py_adamw_destroy(ttm_handle h) {
	auto* st = get_opt(h);
	if (st == nullptr) return;
	Py_XDECREF(st->optimizer);
	*st = {};
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static ttm_optimizer_vtable g_adamw_vtable = {
	py_adamw_create,
	py_adamw_step,
	py_adamw_zero_grad,
	py_adamw_get_lr,
	py_adamw_set_lr,
	py_adamw_destroy,
};

/* ============================================================================
 * Registration / teardown (called from python_model.cpp)
 * ========================================================================= */

ttm_error pyOptimizerRegister(const ttm_host_api* host) {
	if (host->register_optimizer == nullptr) return TTM_OK;
	return host->register_optimizer(host->ctx, "adamw", &g_adamw_vtable);
}

void pyOptimizerTeardown() {
	for (auto& st : g_opts) {
		if (st.used) {
			Py_XDECREF(st.optimizer);
			st = {};
		}
	}
}

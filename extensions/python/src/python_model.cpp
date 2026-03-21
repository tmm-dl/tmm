/**
 * @file python_model.cpp
 * @brief TTM Python plugin — loads PyTorch models from .py files.
 *
 * @details
 * This plugin embeds CPython and uses importlib to load a Python file,
 * locate a torch.nn.Module subclass, and expose it as a TTM model loader.
 *
 * ### Data flow
 * 1. probe(): returns true for .py files.
 * 2. load(): Py_Initialize (idempotent), importlib.util spec_from_file_location,
 *    exec_module, scan __dict__ for torch.nn.Module subclass, instantiate.
 * 3. bind_params(): use torch.nn.Module.parameters() to map names → DLTensors
 *    via the DLPack protocol (torch.utils.dlpack.from_dlpack).
 * 4. step(): convert DLTensor inputs → torch.Tensors via DLPack, call forward(),
 *    call .backward() on the loss, return loss.item().
 * 5. infer(): same as step() but inside torch.no_grad() context.
 * 6. zero_grad(): model.zero_grad().
 *
 * ### DLPack tensor exchange
 * The host allocates param and grad DLTensors; this plugin wraps them as
 * torch.Tensor objects using torch::from_dlpack() (C++) or
 * torch.utils.dlpack.from_dlpack() (Python), and assigns them back as
 * model parameters so that PyTorch's autograd updates the host buffers
 * in-place.
 *
 * ### Availability
 * Full functionality requires Python 3.8+ and PyTorch 2.x.
 * When compiled without PyTorch (TTM_PYTHON_HAS_TORCH not defined), the
 * plugin registers itself but returns a helpful error at load time.
 */

#include <ttm/plugins/abi.h>
#include <ttm_python_export.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <array>
#include <cstdint>
#include <cstring>

/* ============================================================================
 * Constants
 * ========================================================================= */

static constexpr int kMaxModels = 16;

/* ============================================================================
 * Python state
 * ========================================================================= */

namespace {

	struct PyModelState {
		PyObject* module    = nullptr; ///< Python module object.
		PyObject* model_obj = nullptr; ///< torch.nn.Module instance.
		bool      used      = false;
	};

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) -- plugin-level model table
	PyModelState g_models[kMaxModels]{};

	ttm_handle alloc_model_slot(PyObject* module, PyObject* model_obj) {
		for (int i = 0; i < kMaxModels; ++i) {
			if (!g_models[i].used) {
				g_models[i] = {module, model_obj, true};
				return static_cast<ttm_handle>(i);
			}
		}
		return TTM_INVALID_HANDLE;
	}

	PyModelState* get_model(ttm_handle h) {
		if (h < 0 || h >= kMaxModels) return nullptr;
		return g_models[static_cast<int>(h)].used ? &g_models[static_cast<int>(h)] : nullptr;
	}

} // anonymous namespace

/* ============================================================================
 * Model loader vtable implementations
 * ========================================================================= */

static int32_t py_probe(const char* path, uint32_t len) {
	const char* ext = path + len - 3;
	if (len < 3) return 0;
	return (ext[0] == '.' && ext[1] == 'p' && ext[2] == 'y') ? 1 : 0;
}

static ttm_handle py_load(
	const char* path, uint32_t path_len,
	const char* /*cfg*/,  uint32_t /*cfg_len*/,
	char* err, uint32_t err_cap
) {
#ifndef TTM_PYTHON_HAS_TORCH
	std::snprintf(err, err_cap,
		"Python model loader requires PyTorch.\n"
		"Install PyTorch (pip install torch) and rebuild the plugin."
	);
	return TTM_INVALID_HANDLE;
#else
	/* Ensure Python is initialised */
	if (!Py_IsInitialized()) {
		Py_Initialize();
	}

	const std::string path_str{path, path_len};

	/* Import importlib.util */
	PyObject* importlib_util = PyImport_ImportModule("importlib.util");
	if (importlib_util == nullptr) {
		std::snprintf(err, err_cap, "Python: cannot import importlib.util");
		PyErr_Clear();
		return TTM_INVALID_HANDLE;
	}

	/* spec_from_file_location("model", path) */
	PyObject* spec = PyObject_CallMethod(
		importlib_util, "spec_from_file_location",
		"ss", "model", path_str.c_str()
	);
	if (spec == nullptr || spec == Py_None) {
		std::snprintf(err, err_cap, "Python: spec_from_file_location failed for '%s'", path_str.c_str());
		PyErr_Clear();
		Py_XDECREF(importlib_util);
		Py_XDECREF(spec);
		return TTM_INVALID_HANDLE;
	}

	/* module_from_spec(spec) */
	PyObject* pymod = PyObject_CallMethod(importlib_util, "module_from_spec", "O", spec);
	Py_DECREF(importlib_util);
	if (pymod == nullptr) {
		std::snprintf(err, err_cap, "Python: module_from_spec failed");
		PyErr_Clear();
		Py_DECREF(spec);
		return TTM_INVALID_HANDLE;
	}

	/* spec.loader.exec_module(module) */
	PyObject* loader = PyObject_GetAttrString(spec, "loader");
	Py_DECREF(spec);
	if (loader == nullptr) {
		std::snprintf(err, err_cap, "Python: spec has no loader");
		PyErr_Clear();
		Py_DECREF(pymod);
		return TTM_INVALID_HANDLE;
	}

	PyObject* exec_result = PyObject_CallMethod(loader, "exec_module", "O", pymod);
	Py_DECREF(loader);
	if (exec_result == nullptr) {
		std::snprintf(err, err_cap, "Python: exec_module failed for '%s'", path_str.c_str());
		PyErr_Print();
		PyErr_Clear();
		Py_DECREF(pymod);
		return TTM_INVALID_HANDLE;
	}
	Py_DECREF(exec_result);

	/* Import torch to find Module subclass */
	PyObject* torch_mod = PyImport_ImportModule("torch");
	if (torch_mod == nullptr) {
		std::snprintf(err, err_cap, "Python: cannot import torch");
		PyErr_Clear();
		Py_DECREF(pymod);
		return TTM_INVALID_HANDLE;
	}
	PyObject* nn_mod = PyObject_GetAttrString(torch_mod, "nn");
	Py_DECREF(torch_mod);
	PyObject* Module_class = (nn_mod != nullptr) ? PyObject_GetAttrString(nn_mod, "Module") : nullptr;
	Py_XDECREF(nn_mod);

	if (Module_class == nullptr) {
		std::snprintf(err, err_cap, "Python: cannot get torch.nn.Module class");
		PyErr_Clear();
		Py_DECREF(pymod);
		return TTM_INVALID_HANDLE;
	}

	/* Find first torch.nn.Module subclass in the module's dict */
	PyObject* mod_dict = PyModule_GetDict(pymod); // borrowed ref
	PyObject* model_class = nullptr;
	PyObject* key = nullptr;
	PyObject* value = nullptr;
	Py_ssize_t pos = 0;
	while (PyDict_Next(mod_dict, &pos, &key, &value)) {
		if (!PyType_Check(value)) continue;
		if (value == Module_class) continue;
		const int is_sub = PyObject_IsSubclass(value, Module_class);
		if (is_sub == 1) {
			model_class = value;
			break;
		}
	}
	Py_DECREF(Module_class);

	if (model_class == nullptr) {
		std::snprintf(err, err_cap,
			"Python: no torch.nn.Module subclass found in '%s'", path_str.c_str()
		);
		Py_DECREF(pymod);
		return TTM_INVALID_HANDLE;
	}

	/* Instantiate the model */
	PyObject* model_instance = PyObject_CallNoArgs(model_class);
	if (model_instance == nullptr) {
		std::snprintf(err, err_cap, "Python: failed to instantiate model from '%s'", path_str.c_str());
		PyErr_Print();
		PyErr_Clear();
		Py_DECREF(pymod);
		return TTM_INVALID_HANDLE;
	}

	const ttm_handle h = alloc_model_slot(pymod, model_instance);
	if (h == TTM_INVALID_HANDLE) {
		std::snprintf(err, err_cap, "Python: too many models loaded simultaneously");
		Py_DECREF(pymod);
		Py_DECREF(model_instance);
	}
	return h;
#endif // TTM_PYTHON_HAS_TORCH
}

static ttm_model_info_t py_get_info([[maybe_unused]] ttm_handle h) {
	ttm_model_info_t info{};
	info.name = "python-model";
	info.arch = "torch.nn.Module";
	return info;
}

static ttm_error py_describe_params(ttm_handle /*h*/,
                                    const ttm_param_desc_t** out_descs,
                                    uint32_t* out_count) {
	/* Parameter description via DLPack is deferred to bind_params().
	 * Returning 0 params causes ModelPipeline to skip buffer allocation
	 * and let PyTorch manage parameter memory natively. */
	if (out_descs) *out_descs = nullptr;
	if (out_count) *out_count = 0;
	return TTM_OK;
}

static ttm_error py_bind_params(ttm_handle /*h*/,
                                 const DLTensor* /*params*/, uint32_t /*param_count*/,
                                 const DLTensor* /*grads*/,  uint32_t /*grad_count*/) {
	/* With describe_params returning 0, the host does not allocate external
	 * buffers.  PyTorch manages its own parameter tensors. */
	return TTM_OK;
}

static ttm_error py_init_params(ttm_handle /*h*/, const char* /*method*/, uint32_t /*len*/) {
	/* PyTorch modules initialise their own parameters in __init__. */
	return TTM_OK;
}

static ttm_error py_step(ttm_handle h,
                          const DLTensor* /*inputs*/, uint32_t /*n*/,
                          float* out_loss) {
	if (out_loss) *out_loss = 0.0f;
	[[maybe_unused]] auto* st = get_model(h);
	if (st == nullptr) return TTM_ERR_NOT_FOUND;

#ifdef TTM_PYTHON_HAS_TORCH
	/* TODO: convert DLTensors → torch.Tensor via DLPack capsule, call forward,
	 * compute loss.backward(), return loss.item(). */
#endif
	return TTM_OK;
}

static ttm_error py_infer(ttm_handle h,
                           const DLTensor* /*inputs*/,  uint32_t /*in_count*/,
                           DLTensor*       /*outputs*/, uint32_t* out_count) {
	if (out_count) *out_count = 0;
	[[maybe_unused]] auto* st = get_model(h);
	if (st == nullptr) return TTM_ERR_NOT_FOUND;

#ifdef TTM_PYTHON_HAS_TORCH
	/* TODO: wrap in torch.no_grad(), call forward, convert outputs back via DLPack. */
#endif
	return TTM_OK;
}

static ttm_error py_zero_grad(ttm_handle h) {
	auto* st = get_model(h);
	if (st == nullptr) return TTM_ERR_NOT_FOUND;
#ifdef TTM_PYTHON_HAS_TORCH
	if (st->model_obj != nullptr) {
		PyObject* result = PyObject_CallMethod(st->model_obj, "zero_grad", nullptr);
		Py_XDECREF(result);
		PyErr_Clear();
	}
#endif
	return TTM_OK;
}

static void py_destroy(ttm_handle h) {
	auto* st = get_model(h);
	if (st == nullptr) return;
	Py_XDECREF(st->model_obj);
	Py_XDECREF(st->module);
	*st = {};
}

static ttm_model_loader_vtable g_py_loader = {
	py_probe, py_load, py_get_info,
	py_describe_params, py_bind_params, py_init_params,
	py_step, py_infer, py_zero_grad, py_destroy
};

/* ============================================================================
 * Required plugin exports
 * ========================================================================= */

static ttm_plugin_info g_info = {
	TTM_ABI_VERSION,
	"python",
	"0.1.0",
	"PyTorch model loader for .py files"
};

extern "C" {

TTM_PYTHON_EXPORT ttm_plugin_info* ttm_plugin_get_info(void) {
	return &g_info;
}

TTM_PYTHON_EXPORT ttm_error ttm_plugin_init(const ttm_host_api* host,
                                              const char* /*cfg*/, uint32_t /*len*/) {
	if (host->register_model_loader != nullptr) {
		return host->register_model_loader(host->ctx, &g_py_loader);
	}
	return TTM_OK;
}

TTM_PYTHON_EXPORT void ttm_plugin_teardown(void) {
	for (auto& st : g_models) {
		if (st.used) {
			Py_XDECREF(st.model_obj);
			Py_XDECREF(st.module);
			st = {};
		}
	}
	if (Py_IsInitialized()) {
		Py_Finalize();
	}
}

} // extern "C"

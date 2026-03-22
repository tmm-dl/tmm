/**
 * @file python_model.cpp
 * @brief TMM Python plugin — loads PyTorch models from .py files.
 *
 * @details
 * This plugin embeds CPython and uses importlib to load a Python file,
 * locate a torch.nn.Module subclass, and expose it as a TMM model loader.
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
 * When compiled without PyTorch (TMM_PYTHON_HAS_TORCH not defined), the
 * plugin registers itself but returns a helpful error at load time.
 */

#include "python_plugin.hpp"

#include <tmm/plugins/abi.h>
#include <tmm_python_export.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef TMM_PYTHON_HAS_TORCH
#include <dlpack/dlpack.h>
#endif

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#ifndef _WIN32
#include <dlfcn.h>
#endif

/* ============================================================================
 * Constants
 * ========================================================================= */

static constexpr int kMaxModels = 16;

/* ============================================================================
 * Host API reference — stored during tmm_plugin_init for Python log bridge.
 * ========================================================================= */

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static tmm_host_api g_hostApi{};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool g_stdoutRedirected = false;

/// C function exposed to Python as _tmm_host.log(level: int, msg: str).
static PyObject* pyHostLog(PyObject* /*self*/, PyObject* args) {
	int level = 0;
	const char* msg = nullptr;
	if (!PyArg_ParseTuple(args, "is", &level, &msg))
		return nullptr;
	if (g_hostApi.log != nullptr) {
		g_hostApi.log(g_hostApi.ctx, static_cast<tmm_log_level>(level), msg, static_cast<uint32_t>(std::strlen(msg)));
	}
	Py_RETURN_NONE;
}

static PyMethodDef kTmmHostMethods[] = {
		{"log", pyHostLog, METH_VARARGS, "log(level, msg) — route a message via TMM host logger"},
		{nullptr, nullptr, 0, nullptr},
};
static PyModuleDef kTmmHostModuleDef = {
		PyModuleDef_HEAD_INIT, "_tmm_host", nullptr, -1, kTmmHostMethods, nullptr, nullptr, nullptr, nullptr,
};

/// Inject _tmm_host into sys.modules and redirect sys.stdout/sys.stderr.
/// Must be called after Py_Initialize().
static void installPythonLogBridge() {
	if (g_stdoutRedirected)
		return;
	if (!Py_IsInitialized())
		return;

	PyObject* mod = PyModule_Create(&kTmmHostModuleDef);
	if (mod == nullptr) {
		PyErr_Clear();
		return;
	}
	PyObject* sysModules = PyImport_GetModuleDict();
	PyDict_SetItemString(sysModules, "_tmm_host", mod);
	Py_DECREF(mod);

	/* Redirect sys.stdout and sys.stderr through the host logger. */
	PyRun_SimpleString(R"py(
import sys, _tmm_host

class _TmmWriter:
    """Routes Python stdout/stderr into the TMM host logger (e.g. console-ui Logs panel)."""
    def __init__(self, level):
        self._level = level
        self._buf   = ""
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                _tmm_host.log(self._level, line)
    def flush(self):
        if self._buf:
            _tmm_host.log(self._level, self._buf)
            self._buf = ""
    def isatty(self):
        return False
    def fileno(self):
        raise OSError("_TmmWriter has no file descriptor")

sys.stdout = _TmmWriter(2)  # TMM_LOG_INFO
sys.stderr = _TmmWriter(3)  # TMM_LOG_WARN
)py");
	PyErr_Clear(); // ignore any errors from the redirect (best-effort)
	g_stdoutRedirected = true;
}

/* ============================================================================
 * Python state
 * ========================================================================= */

namespace {

	struct PyModelState {
		PyObject* module = nullptr;	   ///< Python module object.
		PyObject* model_obj = nullptr; ///< torch.nn.Module instance.
		bool used = false;
	};

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) -- plugin-level model table
	PyModelState g_models[kMaxModels]{};

	tmm_handle alloc_model_slot(PyObject* module, PyObject* model_obj) {
		for (int i = 0; i < kMaxModels; ++i) {
			if (!g_models[i].used) {
				g_models[i] = {module, model_obj, true};
				return static_cast<tmm_handle>(i);
			}
		}
		return TMM_INVALID_HANDLE;
	}

	PyModelState* get_model(tmm_handle h) {
		if (h < 0 || h >= kMaxModels)
			return nullptr;
		return g_models[static_cast<int>(h)].used ? &g_models[static_cast<int>(h)] : nullptr;
	}

} // anonymous namespace

// Non-static accessor for python_optimizer.cpp (declared in python_plugin.hpp).
PyObject* py_get_model_obj(tmm_handle h) {
	const auto* st =
			(h >= 0 && h < kMaxModels && g_models[static_cast<int>(h)].used) ? &g_models[static_cast<int>(h)] : nullptr;
	return st ? st->model_obj : nullptr;
}

/* ============================================================================
 * DLPack helpers
 * ========================================================================= */

#ifdef TMM_PYTHON_HAS_TORCH

/// Heap-allocated DLManagedTensor that copies the shape from a DLTensor.
/// Freed by managed_dlpack_deleter() which PyTorch calls when it releases the tensor.
struct OwnedDLManaged {
	DLManagedTensor mgd;
	int64_t shape_buf[8]; // supports up to 8-D tensors
};

static void managed_dlpack_deleter(DLManagedTensor* p) {
	// `mgd` is the first member of OwnedDLManaged → reinterpret_cast is safe.
	// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
	delete reinterpret_cast<OwnedDLManaged*>(p);
}

/// Called when the PyCapsule is GC-collected without PyTorch having consumed it.
/// Prevents memory leak on error paths.
static void capsule_dlpack_destructor(PyObject* cap) {
	void* ptr = PyCapsule_GetPointer(cap, "dltensor");
	if (ptr != nullptr) {
		auto* mgd = static_cast<DLManagedTensor*>(ptr);
		if (mgd->deleter != nullptr) {
			mgd->deleter(mgd);
		}
	}
}

/// Wrap a host-owned DLTensor as a "dltensor" PyCapsule for torch.utils.dlpack.from_dlpack().
/// PyTorch takes ownership via the DLManagedTensor::deleter.
static PyObject* make_dlpack_capsule(const DLTensor& t) {
	auto* owned = new OwnedDLManaged{};
	// Copy shape into embedded buffer (host's shape array may not outlive this call)
	for (int i = 0; i < t.ndim && i < 8; ++i) {
		owned->shape_buf[i] = t.shape[i];
	}
	owned->mgd.dl_tensor = t;
	owned->mgd.dl_tensor.shape = owned->shape_buf;
	owned->mgd.dl_tensor.strides = nullptr; // contiguous
	owned->mgd.manager_ctx = nullptr;
	owned->mgd.deleter = managed_dlpack_deleter;
	return PyCapsule_New(&owned->mgd, "dltensor", capsule_dlpack_destructor);
}

/// Convert n DLTensors to a Python list of torch.Tensors via DLPack.
/// Returns a new reference list, or nullptr on error (sets Python exception).
static PyObject* tensors_to_torch(const DLTensor* inputs, uint32_t n) {
	PyObject* dlpack_mod = PyImport_ImportModule("torch.utils.dlpack");
	if (dlpack_mod == nullptr)
		return nullptr;

	PyObject* lst = PyList_New(static_cast<Py_ssize_t>(n));
	for (uint32_t i = 0; i < n; ++i) {
		PyObject* cap = make_dlpack_capsule(inputs[i]);
		PyObject* tensor = PyObject_CallMethod(dlpack_mod, "from_dlpack", "O", cap);
		Py_DECREF(cap);
		if (tensor == nullptr) {
			Py_DECREF(dlpack_mod);
			Py_DECREF(lst);
			return nullptr;
		}
		PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), tensor); // steals ref
	}
	Py_DECREF(dlpack_mod);
	return lst;
}

/// Call model(input_ids=..., attention_mask=..., labels=...) and return output.
/// `tensors` must have at least 2 elements; element 2 is optional labels.
static PyObject* call_model_forward(PyObject* model_obj, PyObject* tensors) {
	const Py_ssize_t ntensors = PyList_GET_SIZE(tensors);

	PyObject* input_ids = PyList_GET_ITEM(tensors, 0);									// borrowed
	PyObject* attention_mask = (ntensors >= 2) ? PyList_GET_ITEM(tensors, 1) : Py_None; // borrowed

	PyObject* kwargs = PyDict_New();
	PyDict_SetItemString(kwargs, "input_ids", input_ids);
	PyDict_SetItemString(kwargs, "attention_mask", attention_mask);

	if (ntensors >= 3) {
		PyObject* raw_labels = PyList_GET_ITEM(tensors, 2); // borrowed [B,1]
		// Squeeze trailing dim-1 → [B] as DistilBERT's forward() expects
		PyObject* labels = PyObject_CallMethod(raw_labels, "squeeze", "i", -1);
		if (labels == nullptr) {
			PyErr_Clear();
			labels = raw_labels;
			Py_INCREF(labels);
		}
		PyDict_SetItemString(kwargs, "labels", labels);
		Py_DECREF(labels);
	}

	PyObject* empty_args = PyTuple_New(0);
	PyObject* output = PyObject_Call(model_obj, empty_args, kwargs);
	Py_DECREF(empty_args);
	Py_DECREF(kwargs);
	return output;
}

#endif // TMM_PYTHON_HAS_TORCH

/* ============================================================================
 * Model loader vtable implementations
 * ========================================================================= */

static int32_t py_probe(const char* path, uint32_t len) {
	const char* ext = path + len - 3;
	if (len < 3)
		return 0;
	return (ext[0] == '.' && ext[1] == 'p' && ext[2] == 'y') ? 1 : 0;
}

static tmm_handle
py_load(const char* path, uint32_t path_len, const char* /*cfg*/, uint32_t /*cfg_len*/, char* err, uint32_t err_cap) {
#ifndef TMM_PYTHON_HAS_TORCH
	std::snprintf(
			err, err_cap,
			"Python model loader requires PyTorch.\n"
			"Install PyTorch (pip install torch) and rebuild the plugin."
	);
	return TMM_INVALID_HANDLE;
#else
	/* Ensure Python is initialised */
	if (!Py_IsInitialized()) {
#ifndef _WIN32
		// Re-open libpython with RTLD_GLOBAL so Python extension modules
		// (e.g. _ctypes, torch) can resolve PyXxx symbols at dlopen time.
		// Without this, embedded Python loaded inside a MODULE shared library
		// does not expose its symbols globally, breaking C-extension imports.
		void* pylib = dlopen("libpython3.12.so.1.0", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
		if (!pylib) {
			pylib = dlopen("libpython3.12.so.1.0", RTLD_NOW | RTLD_GLOBAL);
		}
		if (!pylib) {
			pylib = dlopen("libpython3.12.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
		}
		// pylib intentionally not closed — we want the RTLD_GLOBAL flag to persist.
#endif
		Py_Initialize();
	}
	installPythonLogBridge();

	const std::string path_str{path, path_len};

	/* Import importlib.util */
	PyObject* importlib_util = PyImport_ImportModule("importlib.util");
	if (importlib_util == nullptr) {
		std::snprintf(err, err_cap, "Python: cannot import importlib.util");
		PyErr_Clear();
		return TMM_INVALID_HANDLE;
	}

	/* spec_from_file_location("model", path) */
	PyObject* spec = PyObject_CallMethod(importlib_util, "spec_from_file_location", "ss", "model", path_str.c_str());
	if (spec == nullptr || spec == Py_None) {
		std::snprintf(err, err_cap, "Python: spec_from_file_location failed for '%s'", path_str.c_str());
		PyErr_Clear();
		Py_XDECREF(importlib_util);
		Py_XDECREF(spec);
		return TMM_INVALID_HANDLE;
	}

	/* module_from_spec(spec) */
	PyObject* pymod = PyObject_CallMethod(importlib_util, "module_from_spec", "O", spec);
	Py_DECREF(importlib_util);
	if (pymod == nullptr) {
		std::snprintf(err, err_cap, "Python: module_from_spec failed");
		PyErr_Clear();
		Py_DECREF(spec);
		return TMM_INVALID_HANDLE;
	}

	/* spec.loader.exec_module(module) */
	PyObject* loader = PyObject_GetAttrString(spec, "loader");
	Py_DECREF(spec);
	if (loader == nullptr) {
		std::snprintf(err, err_cap, "Python: spec has no loader");
		PyErr_Clear();
		Py_DECREF(pymod);
		return TMM_INVALID_HANDLE;
	}

	PyObject* exec_result = PyObject_CallMethod(loader, "exec_module", "O", pymod);
	Py_DECREF(loader);
	if (exec_result == nullptr) {
		std::snprintf(err, err_cap, "Python: exec_module failed for '%s'", path_str.c_str());
		PyErr_Print();
		PyErr_Clear();
		Py_DECREF(pymod);
		return TMM_INVALID_HANDLE;
	}
	Py_DECREF(exec_result);

	/* Import torch to find Module subclass */
	PyObject* torch_mod = PyImport_ImportModule("torch");
	if (torch_mod == nullptr) {
		std::snprintf(err, err_cap, "Python: cannot import torch");
		PyErr_Clear();
		Py_DECREF(pymod);
		return TMM_INVALID_HANDLE;
	}
	PyObject* nn_mod = PyObject_GetAttrString(torch_mod, "nn");
	Py_DECREF(torch_mod);
	PyObject* Module_class = (nn_mod != nullptr) ? PyObject_GetAttrString(nn_mod, "Module") : nullptr;
	Py_XDECREF(nn_mod);

	if (Module_class == nullptr) {
		std::snprintf(err, err_cap, "Python: cannot get torch.nn.Module class");
		PyErr_Clear();
		Py_DECREF(pymod);
		return TMM_INVALID_HANDLE;
	}

	/* Find first torch.nn.Module subclass DEFINED in this module's dict.
	 * We filter by __module__ == "model" to skip imported classes
	 * (e.g. DistilBertForSequenceClassification imported at module top-level). */
	PyObject* mod_dict = PyModule_GetDict(pymod); // borrowed ref
	PyObject* model_class = nullptr;
	PyObject* key = nullptr;
	PyObject* value = nullptr;
	Py_ssize_t pos = 0;
	while (PyDict_Next(mod_dict, &pos, &key, &value)) {
		if (!PyType_Check(value))
			continue;
		if (value == Module_class)
			continue;
		// Only consider classes defined in this module, not imported ones
		PyObject* cls_module = PyObject_GetAttrString(value, "__module__");
		bool local = false;
		if (cls_module != nullptr) {
			const char* cm = PyUnicode_AsUTF8(cls_module);
			local = (cm != nullptr && std::strcmp(cm, "model") == 0);
			Py_DECREF(cls_module);
		} else {
			PyErr_Clear();
		}
		if (!local)
			continue;
		const int is_sub = PyObject_IsSubclass(value, Module_class);
		if (is_sub == 1) {
			model_class = value;
			break;
		}
	}
	Py_DECREF(Module_class);

	if (model_class == nullptr) {
		std::snprintf(err, err_cap, "Python: no torch.nn.Module subclass found in '%s'", path_str.c_str());
		Py_DECREF(pymod);
		return TMM_INVALID_HANDLE;
	}

	/* Instantiate the model */
	PyObject* model_instance = PyObject_CallNoArgs(model_class);
	if (model_instance == nullptr) {
		std::snprintf(err, err_cap, "Python: failed to instantiate model from '%s'", path_str.c_str());
		PyErr_Print();
		PyErr_Clear();
		Py_DECREF(pymod);
		return TMM_INVALID_HANDLE;
	}

	const tmm_handle h = alloc_model_slot(pymod, model_instance);
	if (h == TMM_INVALID_HANDLE) {
		std::snprintf(err, err_cap, "Python: too many models loaded simultaneously");
		Py_DECREF(pymod);
		Py_DECREF(model_instance);
	}
	return h;
#endif // TMM_PYTHON_HAS_TORCH
}

static tmm_model_info_t py_get_info([[maybe_unused]] tmm_handle h) {
	tmm_model_info_t info{};
	info.name = "python-model";
	info.arch = "torch.nn.Module";
	return info;
}

static tmm_error py_describe_params(tmm_handle /*h*/, const tmm_param_desc_t** out_descs, uint32_t* out_count) {
	/* Parameter description via DLPack is deferred to bind_params().
	 * Returning 0 params causes ModelPipeline to skip buffer allocation
	 * and let PyTorch manage parameter memory natively. */
	if (out_descs)
		*out_descs = nullptr;
	if (out_count)
		*out_count = 0;
	return TMM_OK;
}

static tmm_error py_bind_params(
		tmm_handle /*h*/, const DLTensor* /*params*/, uint32_t /*param_count*/, const DLTensor* /*grads*/,
		uint32_t /*grad_count*/
) {
	/* With describe_params returning 0, the host does not allocate external
	 * buffers.  PyTorch manages its own parameter tensors. */
	return TMM_OK;
}

static tmm_error py_init_params(tmm_handle /*h*/, const char* /*method*/, uint32_t /*len*/) {
	/* PyTorch modules initialise their own parameters in __init__. */
	return TMM_OK;
}

static tmm_error py_step(tmm_handle h, const DLTensor* inputs, uint32_t n, float* out_loss) {
	if (out_loss)
		*out_loss = 0.0f;
	auto* st = get_model(h);
	if (st == nullptr)
		return TMM_ERR_NOT_FOUND;

#ifdef TMM_PYTHON_HAS_TORCH
	if (n < 1)
		return TMM_ERR_ARGS;

	// 1. Convert DLTensors → torch.Tensors via DLPack
	PyObject* tensors = tensors_to_torch(inputs, n);
	if (tensors == nullptr) {
		PyErr_Print();
		PyErr_Clear();
		return TMM_ERR_IO;
	}

	// 2. Forward pass
	PyObject* output = call_model_forward(st->model_obj, tensors);
	Py_DECREF(tensors);
	if (output == nullptr) {
		if (PyErr_ExceptionMatches(PyExc_KeyboardInterrupt)) {
			PyErr_Clear();
			return TMM_ERR_INTERRUPTED;
		}
		PyErr_Print();
		PyErr_Clear();
		return TMM_ERR_IO;
	}

	// 3. loss.backward()
	PyObject* loss_tensor = PyObject_GetAttrString(output, "loss");
	Py_DECREF(output);
	if (loss_tensor == nullptr || loss_tensor == Py_None) {
		Py_XDECREF(loss_tensor);
		PyErr_Clear();
		return TMM_OK; // no loss (inference-only model?)
	}
	PyObject* bwd = PyObject_CallMethod(loss_tensor, "backward", nullptr);
	Py_XDECREF(bwd);
	if (PyErr_Occurred()) {
		PyErr_Print();
		PyErr_Clear();
	}

	// 4. *out_loss = loss.item()
	PyObject* loss_val = PyObject_CallMethod(loss_tensor, "item", nullptr);
	Py_DECREF(loss_tensor);
	if (loss_val != nullptr) {
		if (out_loss)
			*out_loss = static_cast<float>(PyFloat_AsDouble(loss_val));
		Py_DECREF(loss_val);
	}
	if (PyErr_ExceptionMatches(PyExc_KeyboardInterrupt)) {
		PyErr_Clear();
		return TMM_ERR_INTERRUPTED;
	}
	PyErr_Clear(); // swallow any remaining python error

	// Run Python GC to collect cyclic garbage from the computation graph.
	// Without this, short-lived activation tensors held in reference cycles
	// accumulate until CPython's generational collector runs (which may be
	// infrequent during heavy C-extension use), causing unbounded memory growth.
	static int s_step_count = 0;
	if ((++s_step_count & 0x1F) == 0) { // every 32 steps
		PyObject* gc = PyImport_ImportModule("gc");
		if (gc != nullptr) {
			PyObject* r = PyObject_CallMethod(gc, "collect", nullptr);
			Py_XDECREF(r);
			Py_DECREF(gc);
		}
		PyErr_Clear();
	}
#endif
	return TMM_OK;
}

// thread_local float so py_infer can pass validation loss back to the caller
// via outputs[0].data without allocating heap memory per batch.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local float g_infer_loss = 0.0f;

static tmm_error
py_infer(tmm_handle h, const DLTensor* inputs, uint32_t in_count, DLTensor* outputs, uint32_t* out_count) {
	if (out_count)
		*out_count = 0;
	auto* st = get_model(h);
	if (st == nullptr)
		return TMM_ERR_NOT_FOUND;

#ifdef TMM_PYTHON_HAS_TORCH
	if (in_count < 1)
		return TMM_ERR_ARGS;

	// Disable gradient computation for inference
	PyObject* torch_mod = PyImport_ImportModule("torch");
	if (torch_mod == nullptr) {
		PyErr_Print();
		PyErr_Clear();
		return TMM_ERR_IO;
	}
	PyObject* no_grad_cls = PyObject_GetAttrString(torch_mod, "no_grad");
	Py_DECREF(torch_mod);
	if (no_grad_cls == nullptr) {
		PyErr_Print();
		PyErr_Clear();
		return TMM_ERR_IO;
	}
	PyObject* no_grad_ctx = PyObject_CallObject(no_grad_cls, nullptr);
	Py_DECREF(no_grad_cls);
	if (no_grad_ctx == nullptr) {
		PyErr_Print();
		PyErr_Clear();
		return TMM_ERR_IO;
	}
	PyObject* enter_result = PyObject_CallMethod(no_grad_ctx, "__enter__", nullptr);
	Py_XDECREF(enter_result);
	if (PyErr_Occurred()) {
		PyErr_Print();
		PyErr_Clear();
		Py_DECREF(no_grad_ctx);
		return TMM_ERR_IO;
	}

	// Convert DLTensors → torch.Tensors and run forward
	PyObject* tensors = tensors_to_torch(inputs, in_count);
	PyObject* output = (tensors != nullptr) ? call_model_forward(st->model_obj, tensors) : nullptr;
	Py_XDECREF(tensors);

	// Exit no_grad context (always, even on error)
	PyObject* py_none = Py_None;
	PyObject* exit_result = PyObject_CallMethod(no_grad_ctx, "__exit__", "OOO", py_none, py_none, py_none);
	Py_XDECREF(exit_result);
	Py_DECREF(no_grad_ctx);

	if (output == nullptr) {
		PyErr_Print();
		PyErr_Clear();
		return TMM_ERR_IO;
	}

	// Extract validation loss → store in thread_local so caller can memcpy it
	PyObject* loss_tensor = PyObject_GetAttrString(output, "loss");
	Py_DECREF(output);
	if (loss_tensor != nullptr && loss_tensor != Py_None) {
		PyObject* loss_val = PyObject_CallMethod(loss_tensor, "item", nullptr);
		Py_DECREF(loss_tensor);
		if (loss_val != nullptr) {
			g_infer_loss = static_cast<float>(PyFloat_AsDouble(loss_val));
			Py_DECREF(loss_val);
			if (outputs != nullptr && out_count != nullptr && *out_count >= 1) {
				DLTensor& t = outputs[0];
				t.data = &g_infer_loss;
				t.device = {kDLCPU, 0};
				t.ndim = 0; // scalar
				t.dtype = {kDLFloat, 32, 1};
				t.shape = nullptr;
				*out_count = 1;
			}
		}
	} else {
		Py_XDECREF(loss_tensor);
	}
	PyErr_Clear();
#endif
	return TMM_OK;
}

static tmm_error py_zero_grad(tmm_handle h) {
	auto* st = get_model(h);
	if (st == nullptr)
		return TMM_ERR_NOT_FOUND;
#ifdef TMM_PYTHON_HAS_TORCH
	if (st->model_obj != nullptr) {
		PyObject* result = PyObject_CallMethod(st->model_obj, "zero_grad", nullptr);
		Py_XDECREF(result);
		PyErr_Clear();
	}
#endif
	return TMM_OK;
}

static void py_destroy(tmm_handle h) {
	auto* st = get_model(h);
	if (st == nullptr)
		return;
	Py_XDECREF(st->model_obj);
	Py_XDECREF(st->module);
	*st = {};
}

static tmm_model_loader_vtable g_py_loader = {py_probe,		  py_load, py_get_info, py_describe_params, py_bind_params,
											  py_init_params, py_step, py_infer,	py_zero_grad,		py_destroy};

/* ============================================================================
 * Required plugin exports
 * ========================================================================= */

static tmm_plugin_info g_info = {TMM_ABI_VERSION, "python", "0.1.0", "PyTorch model loader for .py files"};

extern "C" {

TMM_PYTHON_EXPORT tmm_plugin_info* tmm_plugin_get_info(void) { return &g_info; }

TMM_PYTHON_EXPORT tmm_error tmm_plugin_init(const tmm_host_api* host, const char* /*cfg*/, uint32_t /*len*/) {
	if (host != nullptr)
		g_hostApi = *host;
	if (host->register_model_loader != nullptr) {
		const tmm_error rc = host->register_model_loader(host->ctx, &g_py_loader);
		if (rc != TMM_OK)
			return rc;
	}
	{
		const tmm_error rc = pyOptimizerRegister(host);
		if (rc != TMM_OK)
			return rc;
	}
	return pyTransformRegister(host);
}

TMM_PYTHON_EXPORT void tmm_plugin_teardown(void) {
	pyOptimizerTeardown();
	pyTransformTeardown();
	for (auto& st : g_models) {
		if (st.used) {
			Py_XDECREF(st.model_obj);
			Py_XDECREF(st.module);
			st = {};
		}
	}
	g_stdoutRedirected = false;
	g_hostApi = {};
	if (Py_IsInitialized()) {
		Py_Finalize();
	}
}

} // extern "C"

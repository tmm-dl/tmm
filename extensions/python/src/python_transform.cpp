/**
 * @file python_transform.cpp
 * @brief TTM Python plugin — "hf-tokenize" Arrow IPC transform.
 *
 * @details
 * Registers the "hf-tokenize" transform vtable with the host.  Each handle
 * wraps a HuggingFace AutoTokenizer instance.
 *
 * ### Config JSON
 * @code{.json}
 * {
 *   "model_name": "distilbert-base-uncased",
 *   "text_col":   "text",
 *   "label_col":  "label",
 *   "max_length": 128
 * }
 * @endcode
 *
 * ### Input Arrow schema
 * Any RecordBatch with a string `text_col` and optional integer `label_col`.
 *
 * ### Output Arrow schema
 * RecordBatch with columns:
 *   - `input_ids`      : FixedSizeList<int32>[max_length]
 *   - `attention_mask` : FixedSizeList<int32>[max_length]
 *   - `labels`         : int64 (or int32 if source is int32)
 *
 * The output IPC buffer is malloc-allocated; the host frees it.
 */

#include "python_plugin.hpp"

#include <ttm/plugins/abi.h>
#include <ttm_python_export.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>

/* ============================================================================
 * Constants
 * ========================================================================= */

static constexpr int kMaxTransforms = 32;

/* ============================================================================
 * Python transform state
 * ========================================================================= */

namespace {

	struct PyTransformState {
		PyObject* tokenizer = nullptr; ///< transformers.AutoTokenizer instance
		PyObject* helper    = nullptr; ///< Python helper module / callable
		bool      used      = false;
	};

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
	PyTransformState g_transforms[kMaxTransforms]{};

	ttm_handle alloc_transform_slot(PyObject* tok, PyObject* helper) {
		for (int i = 0; i < kMaxTransforms; ++i) {
			if (!g_transforms[i].used) {
				g_transforms[i] = {tok, helper, true};
				return static_cast<ttm_handle>(i);
			}
		}
		return TTM_INVALID_HANDLE;
	}

	PyTransformState* get_transform(ttm_handle h) {
		if (h < 0 || h >= kMaxTransforms) return nullptr;
		return g_transforms[static_cast<int>(h)].used
		       ? &g_transforms[static_cast<int>(h)]
		       : nullptr;
	}

} // anonymous namespace

/* ============================================================================
 * Python helper code (embedded)
 *
 * Loaded once via exec() into a fresh module dict.  The helper function
 * apply_tokenize(tokenizer, ipc_bytes, text_col, label_col, max_length)
 * returns the output IPC bytes (Python bytes object).
 * ========================================================================= */

static const char* kHelperCode = R"py(
import sys, io, struct
import pyarrow as pa
import pyarrow.ipc as pa_ipc

def _read_batch(ipc_bytes):
    reader = pa_ipc.open_stream(pa.BufferReader(ipc_bytes))
    return reader.read_next_batch()

def _write_batch(batch):
    sink = io.BytesIO()
    writer = pa_ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue()

def apply_tokenize(tokenizer, ipc_bytes, text_col, label_col, max_length):
    batch = _read_batch(ipc_bytes)

    texts = batch.column(text_col).to_pylist()

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,  # return plain Python lists
    )

    n = len(texts)
    ml = max_length

    ids_flat  = [tok for row in enc["input_ids"]      for tok in row]
    mask_flat = [tok for row in enc["attention_mask"]  for tok in row]

    ids_array  = pa.FixedSizeListArray.from_arrays(
        pa.array(ids_flat,  type=pa.int32()), ml)
    mask_array = pa.FixedSizeListArray.from_arrays(
        pa.array(mask_flat, type=pa.int32()), ml)

    if label_col and label_col in batch.schema.names:
        labels_array = batch.column(label_col).cast(pa.int64())
    else:
        labels_array = pa.array([0] * n, type=pa.int64())

    out_batch = pa.record_batch({
        "input_ids":      ids_array,
        "attention_mask": mask_array,
        "labels":         labels_array,
    })
    return _write_batch(out_batch)
)py";

/* ============================================================================
 * vtable implementation
 * ========================================================================= */

static ttm_handle hf_tokenize_create(const char* config_json, uint32_t config_len) {
	if (!Py_IsInitialized()) {
		Py_Initialize();
	}

	// Parse config JSON using Python's json module (avoids a C++ JSON dep)
	PyObject* json_mod = PyImport_ImportModule("json");
	if (json_mod == nullptr) { PyErr_Print(); return TTM_INVALID_HANDLE; }

	PyObject* cfg_str = PyUnicode_FromStringAndSize(config_json, static_cast<Py_ssize_t>(config_len));
	PyObject* cfg = PyObject_CallMethod(json_mod, "loads", "O", cfg_str);
	Py_DECREF(cfg_str);
	Py_DECREF(json_mod);
	if (cfg == nullptr) { PyErr_Print(); return TTM_INVALID_HANDLE; }

	// Extract fields
	auto get_str = [&](const char* key, const char* def) -> PyObject* {
		PyObject* v = PyDict_GetItemString(cfg, key); // borrowed
		if (v == nullptr || !PyUnicode_Check(v)) {
			return PyUnicode_FromString(def);
		}
		Py_INCREF(v);
		return v;
	};
	auto get_int = [&](const char* key, long def) -> long {
		PyObject* v = PyDict_GetItemString(cfg, key);
		if (v == nullptr || !PyLong_Check(v)) return def;
		return PyLong_AsLong(v);
	};

	PyObject* model_name = get_str("model_name", "distilbert-base-uncased");
	PyObject* text_col   = get_str("text_col",   "text");
	PyObject* label_col  = get_str("label_col",  "label");
	long      max_length = get_int("max_length",  128);
	Py_DECREF(cfg);

	// Load tokenizer via transformers
	PyObject* transformers = PyImport_ImportModule("transformers");
	if (transformers == nullptr) {
		PyErr_Print();
		Py_DECREF(model_name); Py_DECREF(text_col); Py_DECREF(label_col);
		return TTM_INVALID_HANDLE;
	}
	PyObject* auto_tok_cls = PyObject_GetAttrString(transformers, "AutoTokenizer");
	Py_DECREF(transformers);
	if (auto_tok_cls == nullptr) { PyErr_Print(); return TTM_INVALID_HANDLE; }

	PyObject* tokenizer = PyObject_CallMethod(auto_tok_cls, "from_pretrained", "O", model_name);
	Py_DECREF(auto_tok_cls);
	Py_DECREF(model_name);
	if (tokenizer == nullptr) {
		PyErr_Print();
		Py_DECREF(text_col); Py_DECREF(label_col);
		return TTM_INVALID_HANDLE;
	}

	// Compile and exec the helper code into a fresh dict
	PyObject* helper_dict = PyDict_New();
	PyObject* builtins = PyEval_GetBuiltins(); // borrowed
	PyDict_SetItemString(helper_dict, "__builtins__", builtins);

	PyObject* code_str = PyUnicode_FromString(kHelperCode);
	PyObject* code_obj = Py_CompileString(kHelperCode, "<hf_tokenize_helper>", Py_file_input);
	Py_DECREF(code_str);
	if (code_obj == nullptr) {
		PyErr_Print();
		Py_DECREF(tokenizer); Py_DECREF(text_col); Py_DECREF(label_col); Py_DECREF(helper_dict);
		return TTM_INVALID_HANDLE;
	}
	PyObject* exec_result = PyEval_EvalCode(code_obj, helper_dict, helper_dict);
	Py_DECREF(code_obj);
	Py_XDECREF(exec_result);
	if (PyErr_Occurred()) {
		PyErr_Print();
		Py_DECREF(tokenizer); Py_DECREF(text_col); Py_DECREF(label_col); Py_DECREF(helper_dict);
		return TTM_INVALID_HANDLE;
	}

	// Store text_col, label_col, max_length in a tuple alongside helper_dict
	// Pack as (helper_dict, text_col, label_col, max_length_pyint)
	PyObject* ml_obj = PyLong_FromLong(max_length);
	PyObject* state_tuple = PyTuple_Pack(4, helper_dict, text_col, label_col, ml_obj);
	Py_DECREF(helper_dict); Py_DECREF(text_col); Py_DECREF(label_col); Py_DECREF(ml_obj);
	if (state_tuple == nullptr) {
		PyErr_Print();
		Py_DECREF(tokenizer);
		return TTM_INVALID_HANDLE;
	}

	ttm_handle h = alloc_transform_slot(tokenizer, state_tuple);
	if (h == TTM_INVALID_HANDLE) {
		Py_DECREF(tokenizer);
		Py_DECREF(state_tuple);
	}
	return h;
}

static ttm_error hf_tokenize_apply(
	ttm_handle h,
	const void* in_ipc, uint32_t in_len,
	void** out_ipc, uint32_t* out_len
) {
	auto* st = get_transform(h);
	if (st == nullptr) return TTM_ERR_NOT_FOUND;

	// Unpack state: (helper_dict, text_col, label_col, max_length)
	PyObject* helper_dict = PyTuple_GET_ITEM(st->helper, 0); // borrowed
	PyObject* text_col    = PyTuple_GET_ITEM(st->helper, 1); // borrowed
	PyObject* label_col   = PyTuple_GET_ITEM(st->helper, 2); // borrowed
	PyObject* ml_obj      = PyTuple_GET_ITEM(st->helper, 3); // borrowed

	// Build bytes object from IPC buffer
	PyObject* ipc_bytes = PyBytes_FromStringAndSize(
		static_cast<const char*>(in_ipc),
		static_cast<Py_ssize_t>(in_len)
	);
	if (ipc_bytes == nullptr) { PyErr_Print(); return TTM_ERR_IO; }

	// Call apply_tokenize(tokenizer, ipc_bytes, text_col, label_col, max_length)
	PyObject* fn = PyDict_GetItemString(helper_dict, "apply_tokenize"); // borrowed
	if (fn == nullptr) {
		PyErr_Print();
		Py_DECREF(ipc_bytes);
		return TTM_ERR_NOT_FOUND;
	}

	PyObject* result = PyObject_CallFunctionObjArgs(
		fn, st->tokenizer, ipc_bytes, text_col, label_col, ml_obj, nullptr
	);
	Py_DECREF(ipc_bytes);

	if (result == nullptr) {
		PyErr_Print();
		return TTM_ERR_IO;
	}
	if (!PyBytes_Check(result)) {
		Py_DECREF(result);
		return TTM_ERR_IO;
	}

	// Copy result bytes to malloc buffer (host will free via std::free)
	const Py_ssize_t sz  = PyBytes_GET_SIZE(result);
	const char*      src = PyBytes_AS_STRING(result);
	void* buf = std::malloc(static_cast<std::size_t>(sz));
	if (buf == nullptr) {
		Py_DECREF(result);
		return TTM_ERR_OOM;
	}
	std::memcpy(buf, src, static_cast<std::size_t>(sz));
	Py_DECREF(result);

	*out_ipc = buf;
	*out_len = static_cast<uint32_t>(sz);
	return TTM_OK;
}

static void hf_tokenize_destroy(ttm_handle h) {
	auto* st = get_transform(h);
	if (st == nullptr) return;
	Py_XDECREF(st->tokenizer);
	Py_XDECREF(st->helper);
	*st = {};
}

static ttm_transform_vtable g_hf_tokenize_vtable = {
	hf_tokenize_create,
	hf_tokenize_apply,
	hf_tokenize_destroy,
};

/* ============================================================================
 * Public registration / teardown
 * ========================================================================= */

ttm_error pyTransformRegister(const ttm_host_api* host) {
	if (host->register_transform == nullptr) return TTM_OK;
	return host->register_transform(host->ctx, "hf-tokenize", nullptr, &g_hf_tokenize_vtable);
}

void pyTransformTeardown() {
	for (auto& st : g_transforms) {
		if (st.used) {
			Py_XDECREF(st.tokenizer);
			Py_XDECREF(st.helper);
			st = {};
		}
	}
}

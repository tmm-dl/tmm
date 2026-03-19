/**
 * @file nlp_tasks.c
 * @brief TTM nlp-tasks WASM plugin — HuggingFace-compatible NLP task definitions.
 *
 * @details
 * Compiled to wasm32-wasi via WASI SDK.  Registers the "text-classification"
 * ML task type (and its common aliases) with the host PluginManager.
 *
 * The task definition is compatible with the HuggingFace task IDs:
 *   https://huggingface.co/docs/hub/en/datasets-adding#task_categories
 */

#include <stdint.h>
#include <ttm/plugins/abi.h>

/* ============================================================================
 * Host imports (ttm module namespace, resolved by WAMR)
 * ========================================================================= */

__attribute__((import_module("ttm"), import_name("ttm_log")))
extern void ttm_log_impl(uint32_t level, const char* msg, uint32_t len);

__attribute__((import_module("ttm"), import_name("ttm_register_task")))
extern int32_t ttm_register_task_impl(const char* name, const ttm_task_vtable* vt);

/* ============================================================================
 * text-classification task vtable
 * ========================================================================= */

static const char* tc_name(void) {
	return "text-classification";
}

static const char* tc_aliases_arr[] = {"text-clf", "tc", "sentiment-analysis", NULL};
static const char** tc_aliases(void) {
	return tc_aliases_arr;
}

static const char* tc_inputs_arr[] = {"text", NULL};
static const char** tc_inputs(void) {
	return tc_inputs_arr;
}

static const char* tc_label(void) {
	return "label";
}

static const char* tc_metrics_arr[] = {"accuracy", "f1", NULL};
static const char** tc_metrics(void) {
	return tc_metrics_arr;
}

static const char* tc_loss(void) {
	return "cross_entropy";
}

static ttm_task_vtable g_tc_vtable = {
	tc_name,
	tc_aliases,
	tc_inputs,
	tc_label,
	tc_metrics,
	tc_loss
};

/* ============================================================================
 * token-classification task vtable
 * ========================================================================= */

static const char* ner_name(void) {
	return "token-classification";
}

static const char* ner_aliases_arr[] = {"ner", "named-entity-recognition", NULL};
static const char** ner_aliases(void) {
	return ner_aliases_arr;
}

static const char* ner_inputs_arr[] = {"tokens", NULL};
static const char** ner_inputs(void) {
	return ner_inputs_arr;
}

static const char* ner_label(void) {
	return "ner_tags";
}

static const char* ner_metrics_arr[] = {"seqeval", NULL};
static const char** ner_metrics(void) {
	return ner_metrics_arr;
}

static const char* ner_loss(void) {
	return "cross_entropy";
}

static ttm_task_vtable g_ner_vtable = {
	ner_name,
	ner_aliases,
	ner_inputs,
	ner_label,
	ner_metrics,
	ner_loss
};

/* ============================================================================
 * Plugin metadata
 * ========================================================================= */

static ttm_plugin_info g_info = {
	TTM_ABI_VERSION,
	"nlp-tasks",
	"0.1.0",
	"NLP task definitions: text-classification, token-classification"
};

/* ============================================================================
 * Required plugin exports
 * ========================================================================= */

ttm_plugin_info* ttm_plugin_get_info(void) {
	return &g_info;
}

ttm_error ttm_plugin_init(
	const ttm_host_api* host __attribute__((unused)),
	const char* cfg __attribute__((unused)),
	uint32_t len __attribute__((unused))
) {
	ttm_error ret;

	ret = (ttm_error)ttm_register_task_impl("text-classification", &g_tc_vtable);
	if (ret != TTM_OK) {
		return ret;
	}

	ret = (ttm_error)ttm_register_task_impl("token-classification", &g_ner_vtable);
	if (ret != TTM_OK) {
		return ret;
	}

	return TTM_OK;
}

void ttm_plugin_teardown(void) {
	/* Nothing to clean up */
}

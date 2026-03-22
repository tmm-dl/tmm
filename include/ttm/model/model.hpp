/**
 * @file model.hpp
 * @brief C++ model interface and loader abstraction.
 * @ingroup ttm_model
 */

#pragma once

#include <ttm/compat/expected.hpp>
#include <ttm/model/device.hpp>
#include <ttm/model/params.hpp>
#include <ttm/plugins/abi.h>
#include <ttm/trainer/interfaces.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ttm::model {

	/* =========================================================================
	 * ModelInfo — metadata about a loaded model
	 * ====================================================================== */

	/**
	 * @brief Human-readable and diagnostic metadata for a loaded model.
	 *
	 * @details
	 * Populated by the model loader and broadcast to all plugins via
	 * PluginManager::emit_model_loaded() immediately after a successful load.
	 * Plugins that export #ttm_on_model_loaded (e.g. console-ui) receive this
	 * data as a JSON string.
	 *
	 * @ingroup ttm_model
	 */
	struct ModelInfo {
		std::string name;			   ///< Human-readable model name.
		std::string arch;			   ///< Architecture tag, e.g. "GPT2", "ResNet50".
		uint64_t num_parameters = 0;   ///< Total parameter count.
		uint64_t num_trainable = 0;	   ///< Number of trainable parameters.
		uint64_t bytes_on_device = 0;  ///< Memory footprint (params + grads) in bytes.
		std::string input_schema_json; ///< JSON Arrow schema for expected model inputs.
		Device device;				   ///< Device the model lives on.
	};

	/* =========================================================================
	 * IModel — richer model interface used within the model subsystem
	 * ====================================================================== */

	/**
	 * @brief Extended model interface used inside the model subsystem.
	 *
	 * @details
	 * This interface extends ttm::trainer::IModel with model-system-specific
	 * methods (info(), bind_params()).  The Trainer works with the base
	 * ttm::trainer::IModel interface; only the model subsystem itself uses
	 * the richer methods here.
	 *
	 * Concrete implementations:
	 * - ttm::model::ModelPipeline  — the end-to-end pipeline (preprocess → collate → model)
	 *
	 * @see ttm::trainer::IModel   Base interface used by the Trainer
	 * @see ttm::model::ModelPipeline  Primary implementation
	 * @ingroup ttm_model
	 */
	class IModel : public ttm::trainer::IModel {
	public:
		IModel() = default;
		~IModel() override = default;

		IModel(const IModel&) = delete;
		IModel& operator=(const IModel&) = delete;
		IModel(IModel&&) = default;
		IModel& operator=(IModel&&) = default;

		/** @brief Return static metadata for this model. */
		[[nodiscard]] virtual const ModelInfo& info() const = 0;

		/**
		 * @brief Bind host-allocated parameter and gradient buffers.
		 *
		 * @details
		 * Called by ModelPipeline::load() after describe_params() has been
		 * used to allocate the buffers.  The model must not free them.
		 *
		 * @param params  Parameter + gradient buffer pairs (host-owned).
		 */
		virtual void bind_params(std::vector<ParamBuffer>& params) = 0;
	};

	/* =========================================================================
	 * IModelLoader — plugin-provided factory for IModel instances
	 * ====================================================================== */

	/**
	 * @brief Factory that loads a model from a file and returns an IModel.
	 *
	 * @details
	 * Plugins register a loader by calling
	 * `host->register_model_loader(ctx, &g_my_vtable)` during
	 * #ttm_plugin_init.  The host wraps the vtable in a CModelLoaderAdapter
	 * and adds it to the loader registry.
	 *
	 * ### Selection
	 * ModelPipeline::load() iterates all registered loaders in registration
	 * order and calls probe() on each one.  The first loader that returns
	 * `true` is used for the rest of the load sequence.
	 *
	 * @ingroup ttm_model
	 */
	class IModelLoader {
	public:
		IModelLoader() = default;
		virtual ~IModelLoader() = default;

		IModelLoader(const IModelLoader&) = delete;
		IModelLoader& operator=(const IModelLoader&) = delete;
		IModelLoader(IModelLoader&&) = delete;
		IModelLoader& operator=(IModelLoader&&) = delete;

		/**
		 * @brief Return true if this loader can handle the given file.
		 * @param path  Absolute or relative path to the model file.
		 */
		[[nodiscard]] virtual bool probe(std::string_view path) const = 0;

		/**
		 * @brief Load a model and return an IModel instance.
		 *
		 * @details
		 * Implementations must:
		 * 1. Load/parse the model file.
		 * 2. Return a fully-constructed IModel whose info() is populated.
		 *
		 * bind_params() and init_params() are called by ModelPipeline::load()
		 * after this method returns.
		 *
		 * @param path      Path to the model file.
		 * @param cfg_json  Plugin-specific JSON configuration.
		 * @param dev       Target device for the model.
		 * @return Owning pointer to the new model, or an error string.
		 */
		[[nodiscard]] virtual std::expected<std::unique_ptr<IModel>, std::string>
		load(std::string_view path, std::string_view cfg_json, Device dev) = 0;
	};

} // namespace ttm::model

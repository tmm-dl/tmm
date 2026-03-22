/**
 * @file pipeline.hpp
 * @brief ModelPipeline — end-to-end preprocessing → collation → model.
 * @ingroup tmm_model
 */

#pragma once

#include <tmm/conf/config.hpp>
#include <tmm/model/collator.hpp>
#include <tmm/model/model.hpp>
#include <tmm/model/params.hpp>
#include <tmm/model/preprocessor.hpp>
#include <tmm/plugins/abi.h>
#include <tmm/plugins/plugin_manager.hpp>
#include <tmm/compat/expected.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tmm::model {

	/**
	 * @brief End-to-end training pipeline: preprocess → collate → model.
	 *
	 * @details
	 * ModelPipeline implements tmm::trainer::IModel and is therefore what the
	 * Trainer receives as its `model` argument.  Internally it orchestrates:
	 *
	 * 1. **Preprocessors** — zero or more IPreprocessor instances applied in
	 *    order to transform the raw Arrow RecordBatch (tokenisation, feature
	 *    engineering, etc.).
	 * 2. **Collator** — converts the preprocessed RecordBatch to a set of
	 *    DLTensors understood by the underlying model plugin.
	 * 3. **Plugin model** — the actual model (TVM compiled, PyTorch via Python
	 *    embedding, etc.) called through the #tmm_model_loader_vtable C ABI.
	 * 4. **Param store** — host-allocated parameter and gradient buffers bound
	 *    to the model before training begins.
	 *
	 * ### Loading
	 * Use the static factory ModelPipeline::load():
	 * @code{.cpp}
	 * auto dev = tmm::model::Device::from_string(cfg.model.device);
	 * auto pipe = tmm::model::ModelPipeline::load(cfg.model, mgr, dev);
	 * if (!pipe) { std::cerr << pipe.error() << '\n'; return 1; }
	 * auto trainer = tmm::trainer::Trainer(cfg, mgr, std::move(*pipe), train_factory);
	 * @endcode
	 *
	 * ### Parameter lifecycle
	 * load() calls describe_params() on the plugin model to learn the buffer
	 * layout, allocates host memory for params + grads, binds them back into
	 * the model via bind_params(), and then calls init_params("random") (or
	 * "checkpoint:<path>" if cfg.model.path references a checkpoint).
	 *
	 * @ingroup tmm_model
	 */
	class ModelPipeline final : public IModel {
	public:
		ModelPipeline(ModelPipeline&&)            = default;
		ModelPipeline& operator=(ModelPipeline&&) = default;
		~ModelPipeline() override;

		/** @brief Return the underlying plugin model handle (e.g. for optimizer creation). */
		[[nodiscard]] tmm_handle model_handle() const { return handle_; }

		/**
		 * @brief Load a model and set up the full preprocessing pipeline.
		 *
		 * @details
		 * Steps performed:
		 * 1. Find a registered IModelLoader that probe()s true for `cfg.path`.
		 * 2. Call loader->load() to get the underlying IModel.
		 * 3. Query registered preprocessors from the plugin manager for all
		 *    types listed in `cfg_preprocessors`.
		 * 4. Build a default ICollator from the model's input_schema_json.
		 * 5. Allocate param + grad buffers (CPU only for now).
		 * 6. Call bind_params() + init_params("random") on the model.
		 * 7. Call mgr.emitModelLoaded() to broadcast model metadata.
		 *
		 * @param cfg              Model configuration section.
		 * @param preprocessors    Ordered list of preprocessor config entries.
		 * @param mgr              Plugin manager (must outlive the pipeline).
		 * @param dev              Target device.
		 * @return Owning pointer to the pipeline, or an error string.
		 */
		[[nodiscard]] static std::expected<std::unique_ptr<ModelPipeline>, std::string>
		load(
			const conf::ModelConfig&                     cfg,
			const std::vector<conf::PreprocessorEntry>&  preprocessors,
			plugins::PluginManager&                      mgr,
			Device                                       dev
		);

		// ── tmm::trainer::IModel ──────────────────────────────────────────

		[[nodiscard]] std::string_view name() const override;

		/**
		 * @brief Forward + backward pass.
		 * @details Runs preprocessors → collation → plugin model step().
		 */
		trainer::StepOutput step(const trainer::Batch& batch)  override;

		/**
		 * @brief Forward-only pass (validation / inference).
		 * @details Runs preprocessors → collation → plugin model infer().
		 */
		trainer::StepOutput infer(const trainer::Batch& batch) override;

		/** @brief Zero all gradient buffers. */
		void zeroGrad() override;

		// ── tmm::model::IModel ────────────────────────────────────────────

		/** @brief Return static model metadata. */
		[[nodiscard]] const ModelInfo& info() const override { return info_; }

		/** @brief Bind externally-allocated param/grad buffers (no-op after load()). */
		void bind_params(std::vector<ParamBuffer>& params) override;

	private:
		ModelPipeline() = default;

		/** @brief Apply the preprocessor chain and return the transformed batch. */
		[[nodiscard]] std::expected<std::shared_ptr<arrow::RecordBatch>, std::string>
		runPreprocessors(const arrow::RecordBatch& raw) const;

		// Owned resources
		std::unique_ptr<IModel>             inner_;         ///< Plugin-provided model.
		std::unique_ptr<ICollator>          collator_;      ///< Arrow → DLTensor collator.
		std::vector<ParamBuffer>            params_;        ///< Param + grad buffers.
		tmm_handle                          handle_ = TMM_INVALID_HANDLE; ///< Underlying model handle.

		// Owned preprocessors (instantiated with per-entry config at load time)
		std::vector<std::unique_ptr<IPreprocessor>> preprocessors_;

		ModelInfo                           info_;
	};

} // namespace tmm::model

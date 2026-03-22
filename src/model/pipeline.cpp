/**
 * @file pipeline.cpp
 * @brief ModelPipeline implementation.
 *
 * @details
 * ModelPipeline::load() performs the full model-load sequence:
 *  1. Find a registered IModelLoader that accepts cfg.path.
 *  2. Open the model → IModelLoader handle + vtable.
 *  3. Resolve preprocessors from PluginManager.
 *  4. Build a default ICollator.
 *  5. Describe, allocate, and bind parameter/gradient buffers.
 *  6. Initialise parameters ("random" by default).
 *  7. Emit model metadata to all loaded plugins.
 *
 * step() and infer() run preprocessors → collation → plugin vtable call.
 */

#include <ttm/model/collator.hpp>
#include <ttm/model/pipeline.hpp>
#include <ttm/plugins/extension.hpp>
#include <ttm/plugins/plugin_manager.hpp>

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>

#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// Platform-specific aligned allocation
#if defined(_WIN32)
#include <malloc.h>
#define TTM_ALIGNED_ALLOC(align, size) _aligned_malloc((size), (align))
#define TTM_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
static void* ttm_aligned_alloc(std::size_t align, std::size_t size) {
	void* p = nullptr;
	if (::posix_memalign(&p, align, size) != 0)
		return nullptr;
	return p;
}
#define TTM_ALIGNED_ALLOC(align, size) ttm_aligned_alloc((align), (size))
#define TTM_ALIGNED_FREE(ptr) ::free(ptr)
#endif

namespace ttm::model {

	/* =========================================================================
	 * CVtableModel — wraps a plugin's ttm_model_loader_vtable handle
	 * ====================================================================== */

	/**
	 * @brief IModel backed by a plugin-provided ttm_model_loader_vtable.
	 *
	 * @details
	 * Created by ModelPipeline::load(); not directly visible to callers.
	 * The vtable pointer is non-owning (plugin lifetime exceeds this object).
	 */
	class CVtableModel final : public IModel {
	public:
		CVtableModel(plugins::IModelLoader* loader, ttm_handle handle, ModelInfo info)
				: loader_(loader), handle_(handle), info_(std::move(info)) {}

		~CVtableModel() override {
			if (handle_ != TTM_INVALID_HANDLE) {
				loader_->destroy(handle_);
				handle_ = TTM_INVALID_HANDLE;
			}
		}

		CVtableModel(const CVtableModel&) = delete;
		CVtableModel& operator=(const CVtableModel&) = delete;
		CVtableModel(CVtableModel&&) = delete;
		CVtableModel& operator=(CVtableModel&&) = delete;

		[[nodiscard]] std::string_view name() const override { return info_.name; }

		[[nodiscard]] const ModelInfo& info() const override { return info_; }

		void bind_params(std::vector<ParamBuffer>& params) override {
			if (params.empty())
				return;

			std::vector<DLTensor> pTensors;
			std::vector<DLTensor> gTensors;
			pTensors.reserve(params.size());
			gTensors.reserve(params.size());
			for (auto& pb : params) {
				pTensors.push_back(pb.tensor);
				gTensors.push_back(pb.grad);
			}

			loader_->bind_params(
					handle_, pTensors.data(), static_cast<uint32_t>(pTensors.size()), gTensors.data(),
					static_cast<uint32_t>(gTensors.size())
			);
		}

		trainer::StepOutput step(const trainer::Batch& /*batch*/) override {
			// Should not be called directly; ModelPipeline calls step_with_tensors
			return {};
		}

		trainer::StepOutput infer(const trainer::Batch& /*batch*/) override {
			// Should not be called directly; ModelPipeline calls infer_with_tensors
			return {};
		}

		void zero_grad() override { loader_->zero_grad(handle_); }

		// Direct tensor-based call paths used by ModelPipeline ──────────────

		[[nodiscard]] trainer::StepOutput step_with_tensors(const std::vector<DLTensor>& inputs) {
			float loss = 0.0f;
			const auto err = loader_->step(
					handle_, inputs.empty() ? nullptr : inputs.data(), static_cast<uint32_t>(inputs.size()), &loss
			);
			if (err == TTM_ERR_INTERRUPTED)
				return {.loss = 0.0f, .interrupted = true};
			if (err != TTM_OK) {
				std::cerr << "[ttm/pipeline] step() returned error " << err << '\n';
			}
			return {.loss = loss};
		}

		[[nodiscard]] trainer::StepOutput infer_with_tensors(const std::vector<DLTensor>& inputs) {
			// For simple loss-based models, use step with a maximum output buffer.
			// Richer infer() output is accessible via the loader directly.
			constexpr uint32_t kMaxOutputs = 8;
			std::vector<DLTensor> outputs(kMaxOutputs);
			uint32_t out_count = kMaxOutputs;
			const auto err = loader_->infer(
					handle_, inputs.empty() ? nullptr : inputs.data(), static_cast<uint32_t>(inputs.size()),
					outputs.data(), &out_count
			);
			if (err != TTM_OK) {
				// Fallback: try step (many models only implement step)
				return step_with_tensors(inputs);
			}
			// Return loss from first output if available
			float loss = std::numeric_limits<float>::quiet_NaN();
			if (out_count > 0 && outputs[0].data != nullptr) {
				std::memcpy(&loss, outputs[0].data, sizeof(float));
			}
			return {loss};
		}

		[[nodiscard]] ttm_error describe_params(const ttm_param_desc_t** out_descs, uint32_t* out_count) {
			return loader_->describe_params(handle_, out_descs, out_count);
		}

		[[nodiscard]] ttm_error init_params(std::string_view method) { return loader_->init_params(handle_, method); }

		plugins::IModelLoader* loader() const { return loader_; }
		ttm_handle handle() const { return handle_; }

	private:
		plugins::IModelLoader* loader_ = nullptr;
		ttm_handle handle_ = TTM_INVALID_HANDLE;
		ModelInfo info_;
	};

	/* =========================================================================
	 * Helpers
	 * ====================================================================== */

	namespace {

		/// Convert ttm_model_info_t → ModelInfo using info string.
		ModelInfo from_c_info(const ttm_model_info_t& ci, Device dev) {
			ModelInfo m;
			if (ci.name)
				m.name = ci.name;
			if (ci.arch)
				m.arch = ci.arch;
			m.num_parameters = ci.num_parameters;
			m.num_trainable = ci.num_trainable;
			m.bytes_on_device = ci.bytes_on_device;
			if (ci.input_schema_json)
				m.input_schema_json = ci.input_schema_json;
			m.device = dev;
			return m;
		}

		/// Build a JSON model info string for emit_model_loaded().
		std::string build_model_info_json(const ModelInfo& m) {
			return std::format(
					R"({{"name":"{}","arch":"{}","num_parameters":{},"num_trainable":{},"bytes_on_device":{},"device":"{}","device_id":{}}})",
					m.name, m.arch, m.num_parameters, m.num_trainable, m.bytes_on_device,
					[&m]() -> std::string {
						switch (m.device.type) {
						case kDLCPU:
							return "cpu";
						case kDLCUDA:
							return "cuda";
						case kDLMetal:
							return "metal";
						case kDLOpenCL:
							return "opencl";
						default:
							return "cpu";
						}
					}(),
					m.device.id
			);
		}

		constexpr std::size_t kParamAlign = 64; ///< 64-byte alignment (AVX-512 friendly).

		/// Allocate a DLTensor buffer of the given byte count.
		/// Returns nullptr on failure.
		void* alloc_param_buf(std::size_t nbytes) {
			if (nbytes == 0)
				return nullptr;
			return TTM_ALIGNED_ALLOC(kParamAlign, nbytes);
		}

		/// Compute the byte size of a parameter descriptor.
		std::size_t param_byte_size(const ttm_param_desc_t& desc) {
			std::size_t elems = 1;
			for (int32_t d = 0; d < desc.ndim; ++d) {
				elems *= static_cast<std::size_t>(desc.shape[d]);
			}
			return elems * static_cast<std::size_t>(desc.dtype_bits / 8);
		}

	} // anonymous namespace

	/* =========================================================================
	 * TransformPreprocessorAdapter — wraps ttm_transform_vtable as IPreprocessor
	 * ====================================================================== */

	/**
	 * @brief Adapts a plugin-registered ttm_transform_vtable as an IPreprocessor.
	 *
	 * @details
	 * Creates a new transform instance using the per-preprocessor config JSON
	 * (not the empty-config default instance stored in the transform registry).
	 * Serialises the input RecordBatch to Arrow IPC, calls apply(), then
	 * deserialises the output back to a RecordBatch.
	 */
	class TransformPreprocessorAdapter final : public model::IPreprocessor {
	public:
		TransformPreprocessorAdapter(
				std::string type_name, const ttm_transform_vtable& vt, std::string_view config_json
		)
				: name_(std::move(type_name)), vt_(vt), handle_(TTM_INVALID_HANDLE) {
			if (vt_.create != nullptr) {
				handle_ = vt_.create(config_json.data(), static_cast<uint32_t>(config_json.size()));
			}
		}

		~TransformPreprocessorAdapter() override {
			if (handle_ != TTM_INVALID_HANDLE && vt_.destroy != nullptr) {
				vt_.destroy(handle_);
			}
		}

		TransformPreprocessorAdapter(const TransformPreprocessorAdapter&) = delete;
		TransformPreprocessorAdapter& operator=(const TransformPreprocessorAdapter&) = delete;
		TransformPreprocessorAdapter(TransformPreprocessorAdapter&&) = delete;
		TransformPreprocessorAdapter& operator=(TransformPreprocessorAdapter&&) = delete;

		[[nodiscard]] std::string_view name() const override { return name_; }

		[[nodiscard]] std::expected<std::shared_ptr<arrow::RecordBatch>, std::string>
		apply(const arrow::RecordBatch& batch) const override {
			if (handle_ == TTM_INVALID_HANDLE) {
				return std::unexpected(
						std::format("TransformPreprocessorAdapter('{}'): handle is invalid (create failed?)", name_)
				);
			}
			if (vt_.apply == nullptr) {
				return std::unexpected(
						std::format("TransformPreprocessorAdapter('{}'): vtable has no apply() function", name_)
				);
			}

			// 1. Serialize input batch to Arrow IPC stream format
			auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
			{
				auto writer = arrow::ipc::MakeStreamWriter(sink.get(), batch.schema()).ValueOrDie();
				auto st = writer->WriteRecordBatch(batch);
				if (!st.ok()) {
					return std::unexpected(
							std::format(
									"TransformPreprocessorAdapter('{}'): IPC serialise failed: {}", name_, st.ToString()
							)
					);
				}
				st = writer->Close();
				if (!st.ok()) {
					return std::unexpected(
							std::format(
									"TransformPreprocessorAdapter('{}'): IPC writer close failed: {}", name_,
									st.ToString()
							)
					);
				}
			}
			auto in_buf = sink->Finish().ValueOrDie();

			// 2. Call transform
			void* out_raw = nullptr;
			uint32_t out_len = 0;
			const auto err =
					vt_.apply(handle_, in_buf->data(), static_cast<uint32_t>(in_buf->size()), &out_raw, &out_len);
			if (err != TTM_OK) {
				return std::unexpected(
						std::format(
								"TransformPreprocessorAdapter('{}'): apply() returned error {}", name_,
								static_cast<int>(err)
						)
				);
			}

			// 3. Deserialize output — wrap the malloc'd buffer in an owning Arrow
			// Buffer so that RecordBatch column slices (zero-copy IPC) keep the
			// underlying memory alive for as long as the batch is referenced.
			// Arrow's IPC reader takes shared_ptr slices of the input buffer;
			// without this, `out_raw` would be freed when `apply()` returns while
			// the column arrays still hold dangling pointers into it.
			class MallocBuffer final : public arrow::Buffer {
			public:
				MallocBuffer(void* ptr, int64_t size)
						: arrow::Buffer(static_cast<const uint8_t*>(ptr), size), ptr_(ptr) {}
				~MallocBuffer() override { std::free(ptr_); }

			private:
				void* ptr_;
			};
			auto out_buf = std::shared_ptr<arrow::Buffer>(new MallocBuffer(out_raw, static_cast<int64_t>(out_len)));
			// Ownership transferred to out_buf — do not free out_raw separately.
			out_raw = nullptr;

			auto buf_reader = std::make_shared<arrow::io::BufferReader>(out_buf);
			auto reader_result = arrow::ipc::RecordBatchStreamReader::Open(buf_reader);
			if (!reader_result.ok()) {
				return std::unexpected(
						std::format(
								"TransformPreprocessorAdapter('{}'): IPC deserialise failed: {}", name_,
								reader_result.status().ToString()
						)
				);
			}
			auto reader = std::move(reader_result).ValueOrDie();

			std::shared_ptr<arrow::RecordBatch> out_batch;
			auto st = reader->ReadNext(&out_batch);
			if (!st.ok() || out_batch == nullptr) {
				return std::unexpected(
						std::format("TransformPreprocessorAdapter('{}'): no batch in IPC output", name_)
				);
			}
			return out_batch;
		}

	private:
		std::string name_;
		ttm_transform_vtable vt_;
		ttm_handle handle_;
	};

	/* =========================================================================
	 * ModelPipeline::~ModelPipeline
	 * ====================================================================== */

	ModelPipeline::~ModelPipeline() {
		// Free host-allocated param / grad buffers
		for (auto& pb : params_) {
			TTM_ALIGNED_FREE(pb.tensor.data);
			TTM_ALIGNED_FREE(pb.grad.data);
		}
	}

	/* =========================================================================
	 * ModelPipeline::load
	 * ====================================================================== */

	std::expected<std::unique_ptr<ModelPipeline>, std::string> ModelPipeline::load(
			const conf::ModelConfig& cfg, const std::vector<conf::PreprocessorEntry>& preprocessor_entries,
			plugins::PluginManager& mgr, Device dev
	) {
		// 1. Find a loader ────────────────────────────────────────────────────
		plugins::IModelLoader* loader = mgr.find_model_loader(cfg.path);
		if (loader == nullptr) {
			return std::unexpected(
					std::format(
							"ModelPipeline: no registered loader accepts '{}'.\n"
							"  Make sure a plugin implementing ttm_model_loader_vtable is loaded\n"
							"  and its probe() function returns non-zero for this file extension.",
							cfg.path
					)
			);
		}

		// 2. Open the model ───────────────────────────────────────────────────
		auto openResult = loader->open(cfg.path, "{}");
		if (!openResult) {
			return std::unexpected(std::format("ModelPipeline: failed to load '{}': {}", cfg.path, openResult.error()));
		}
		const ttm_handle handle = *openResult;

		// 3. Build ModelInfo from the plugin ──────────────────────────────────
		const ttm_model_info_t ci = loader->get_info(handle);
		ModelInfo info = from_c_info(ci, dev);
		if (info.name.empty())
			info.name = cfg.path;

		// 4. Create the inner model object ────────────────────────────────────
		auto inner = std::make_unique<CVtableModel>(loader, handle, info);

		// 5. Resolve preprocessors from plugin registry ───────────────────────
		std::vector<std::unique_ptr<IPreprocessor>> preprocessors;
		for (const auto& pe : preprocessor_entries) {
			const auto* vt = mgr.find_transform_vtable(pe.type);
			if (vt == nullptr) {
				return std::unexpected(
						std::format(
								"ModelPipeline: no transform registered for '{}'. "
								"Make sure the plugin providing this transform is loaded.",
								pe.type
						)
				);
			}
			const std::string cfg = pe.config.empty() ? "{}" : pe.config;
			preprocessors.push_back(std::make_unique<TransformPreprocessorAdapter>(pe.type, *vt, cfg));
		}

		// 6. Build collator ───────────────────────────────────────────────────
		// Build a minimal schema for the default collator.
		// When a real input_schema_json is present, we'd parse it here.
		auto schema = std::make_shared<arrow::Schema>(arrow::FieldVector{});
		auto collator = make_default_collator(*schema);

		// 7. Describe and allocate parameters ─────────────────────────────────
		const ttm_param_desc_t* descs = nullptr;
		uint32_t n_descs = 0;
		std::vector<ParamBuffer> params;

		const auto desc_err = inner->describe_params(&descs, &n_descs);
		if (desc_err == TTM_OK && descs != nullptr && n_descs > 0) {
			params.resize(n_descs);
			uint64_t total_bytes = 0;
			uint64_t trainable_count = 0;

			for (uint32_t i = 0; i < n_descs; ++i) {
				const auto& desc = descs[i];
				params[i].name = desc.name ? desc.name : "";
				params[i].trainable = desc.trainable != 0;

				const std::size_t nbytes = param_byte_size(desc);
				total_bytes += nbytes;
				if (params[i].trainable)
					++trainable_count;

				// Allocate param tensor
				void* param_buf = alloc_param_buf(nbytes);
				void* grad_buf = alloc_param_buf(nbytes);
				if ((nbytes > 0) && (param_buf == nullptr || grad_buf == nullptr)) {
					TTM_ALIGNED_FREE(param_buf);
					TTM_ALIGNED_FREE(grad_buf);
					return std::unexpected("ModelPipeline: out of memory allocating parameters");
				}
				if (nbytes > 0) {
					std::memset(param_buf, 0, nbytes);
					std::memset(grad_buf, 0, nbytes);
				}

				// Build shape storage (copy from desc)
				auto& tensor = params[i].tensor;
				auto& grad = params[i].grad;

				// We use the shape from desc directly (plugin-owned, valid until destroy)
				tensor.data = param_buf;
				tensor.device = dev.to_dl();
				tensor.ndim = desc.ndim;
				tensor.dtype = {static_cast<uint8_t>(desc.dtype_code), static_cast<uint8_t>(desc.dtype_bits), 1};
				tensor.shape = const_cast<int64_t*>(desc.shape);
				tensor.strides = nullptr;
				tensor.byte_offset = 0;

				grad = tensor; // same layout
				grad.data = grad_buf;
			}

			// Update info with accurate counts
			info.num_parameters = n_descs;
			info.num_trainable = trainable_count;
			info.bytes_on_device = total_bytes * 2; // params + grads

			// 8. Bind params to model ──────────────────────────────────────────
			inner->bind_params(params);

			// 9. Init params (random) ─────────────────────────────────────────
			[[maybe_unused]] auto init_err = inner->init_params("random");
		}

		// 10. Build pipeline object ───────────────────────────────────────────
		auto pipe = std::unique_ptr<ModelPipeline>(new ModelPipeline());
		pipe->handle_ = handle;
		pipe->inner_ = std::move(inner);
		pipe->collator_ = std::move(collator);
		pipe->params_ = std::move(params);
		pipe->preprocessors_ = std::move(preprocessors);
		pipe->info_ = std::move(info);

		// 11. Broadcast model metadata ────────────────────────────────────────
		mgr.emit_model_loaded(build_model_info_json(pipe->info_));

		return pipe;
	}

	/* =========================================================================
	 * Preprocessing helpers
	 * ====================================================================== */

	std::expected<std::shared_ptr<arrow::RecordBatch>, std::string>
	ModelPipeline::run_preprocessors(const arrow::RecordBatch& raw) const {
		// Wrap the raw batch in a shared_ptr without copying — preprocessors
		// that need ownership can copy themselves.
		std::shared_ptr<arrow::RecordBatch> batch{
				const_cast<arrow::RecordBatch*>(&raw), // NOLINT -- non-owning alias
				[](arrow::RecordBatch*) {}			   // no-op deleter
		};
		for (const auto& pp : preprocessors_) {
			auto result = pp->apply(*batch);
			if (!result)
				return std::unexpected(result.error());
			batch = std::move(*result);
		}
		return batch;
	}

	/* =========================================================================
	 * step / infer / zero_grad / name / bind_params
	 * ====================================================================== */

	std::string_view ModelPipeline::name() const { return info_.name; }

	trainer::StepOutput ModelPipeline::step(const trainer::Batch& batch) {
		// 1. Preprocess
		auto processed = run_preprocessors(*batch.data);
		if (!processed) {
			std::cerr << "[ttm/pipeline] preprocessing failed: " << processed.error() << '\n';
			return {};
		}

		// 2. Collate
		auto collated = collator_->collate(**processed);
		if (!collated) {
			std::cerr << "[ttm/pipeline] collation failed: " << collated.error() << '\n';
			return {};
		}

		// 3. Call model
		auto* inner = dynamic_cast<CVtableModel*>(inner_.get());
		if (inner == nullptr)
			return {};
		return inner->step_with_tensors(collated->inputs);
	}

	trainer::StepOutput ModelPipeline::infer(const trainer::Batch& batch) {
		auto processed = run_preprocessors(*batch.data);
		if (!processed) {
			std::cerr << "[ttm/pipeline] preprocessing failed (infer): " << processed.error() << '\n';
			return {};
		}

		auto collated = collator_->collate(**processed);
		if (!collated) {
			std::cerr << "[ttm/pipeline] collation failed (infer): " << collated.error() << '\n';
			return {};
		}

		auto* inner = dynamic_cast<CVtableModel*>(inner_.get());
		if (inner == nullptr)
			return {};
		return inner->infer_with_tensors(collated->inputs);
	}

	void ModelPipeline::zero_grad() {
		if (inner_)
			inner_->zero_grad();
	}

	void ModelPipeline::bind_params(std::vector<ParamBuffer>& params) {
		if (inner_)
			inner_->bind_params(params);
	}

} // namespace ttm::model

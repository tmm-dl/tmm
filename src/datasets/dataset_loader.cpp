/**
 * @file dataset_loader.cpp
 * @brief Dataset loading implementation — Arrow IPC / Parquet multi-file iterator.
 *
 * @details
 * Uses Apache Arrow C++ to read `.arrow` (IPC) and `.parquet` files.  Each
 * file is opened as an Arrow `RandomAccessFile` adapter wrapping an
 * `IByteReader`, then handed to the appropriate Arrow reader.  Multiple shard
 * files are chained into a single DatasetIterator.
 */

#include <ttm/datasets/dataset_loader.hpp>

#include <ttm/datasets/dataset_info.hpp>
#include <ttm/plugins/extension.hpp>

#include <filesystem>
#include <ttm/compat/format.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <ttm/compat/expected.hpp>
#include <utility>
#include <vector>

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/ipc/reader.h>
#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>

namespace ttm::datasets {

	namespace {

		/* =========================================================================
		 * IByteReaderFile — Arrow RandomAccessFile adapter for IByteReader
		 * ====================================================================== */

		/**
		 * @brief Arrow `RandomAccessFile` backed by a `ttm::plugins::IByteReader`.
		 *
		 * @details
		 * Arrow's `RecordBatchFileReader` and Parquet reader both require a
		 * `RandomAccessFile`.  This adapter forwards Arrow I/O calls to the
		 * `IByteReader` interface provided by the plugin system.
		 */
		class IByteReaderFile final : public arrow::io::RandomAccessFile {
		public:
			/**
			 * @param[in] reader   Owning pointer to the IByteReader to adapt.
			 * @param[in] size     Total file size in bytes (required by Arrow).
			 */
			IByteReaderFile(std::unique_ptr<plugins::IByteReader> reader, int64_t size)
					: reader_(std::move(reader)), size_(size) {}

			arrow::Status Close() override {
				closed_ = true;
				return arrow::Status::OK();
			}

			[[nodiscard]] bool closed() const override { return closed_; }

			arrow::Result<int64_t> Tell() const override {
				return arrow::Result<int64_t>(pos_);
			}

			arrow::Status Seek(int64_t pos) override {
				if (!reader_->seekable()) {
					return arrow::Status::IOError("IByteReaderFile::Seek: reader is not seekable");
				}
				const auto result = reader_->seek(static_cast<std::streamoff>(pos), std::ios_base::beg);
				if (static_cast<int64_t>(result) < 0) {
					return arrow::Status::IOError("IByteReaderFile::Seek: seek failed");
				}
				pos_ = pos;
				return arrow::Status::OK();
			}

			arrow::Result<int64_t> GetSize() override {
				return arrow::Result<int64_t>(size_);
			}

			arrow::Result<int64_t> Read(int64_t nbytes, void* out) override {
				// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) -- Arrow API uses void*; IByteReader uses std::byte*; cast is safe
				const auto nRead = reader_->read(reinterpret_cast<std::byte*>(out), static_cast<std::streamsize>(nbytes));
				if (nRead < 0) {
					return arrow::Status::IOError("IByteReaderFile::Read failed");
				}
				pos_ += nRead;
				return arrow::Result<int64_t>(nRead);
			}

			arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override {
				ARROW_ASSIGN_OR_RAISE(auto buf, arrow::AllocateResizableBuffer(nbytes));
				ARROW_ASSIGN_OR_RAISE(const int64_t nRead, Read(nbytes, buf->mutable_data()));
				ARROW_RETURN_NOT_OK(buf->Resize(nRead, false));
				return std::shared_ptr<arrow::Buffer>(std::move(buf));
			}

			arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t pos, int64_t nbytes) override {
				ARROW_RETURN_NOT_OK(Seek(pos));
				return Read(nbytes);
			}

			arrow::Result<int64_t> ReadAt(int64_t pos, int64_t nbytes, void* out) override {
				ARROW_RETURN_NOT_OK(Seek(pos));
				return Read(nbytes, out);
			}

		private:
			std::unique_ptr<plugins::IByteReader> reader_;
			int64_t size_ = 0;
			int64_t pos_  = 0;
			bool    closed_ = false;
		};

		/* =========================================================================
		 * File-size helper
		 * ====================================================================== */

		/**
		 * @brief Determine the byte size of an IByteReader by seeking to the end.
		 * @return File size in bytes, or -1 on failure.
		 */
		int64_t probe_size(plugins::IByteReader& reader) {
			if (!reader.seekable()) {
				return -1;
			}
			const auto end = reader.seek(0, std::ios_base::end);
			if (static_cast<int64_t>(end) < 0) {
				return -1;
			}
			reader.seek(0, std::ios_base::beg);
			return static_cast<int64_t>(end);
		}

		/* =========================================================================
		 * Arrow IPC shard reader
		 * ====================================================================== */

		/**
		 * @brief DatasetIterator for a single Arrow IPC file.
		 */
		class ArrowIpcIterator final : public DatasetIterator {
		public:
			explicit ArrowIpcIterator(std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader)
					: reader_(std::move(reader)), numBatches_(reader_->num_record_batches()) {}

			bool next(std::shared_ptr<arrow::RecordBatch>& out) override {
				if (idx_ >= numBatches_) {
					return false;
				}
				const auto result = reader_->ReadRecordBatch(idx_);
				if (!result.ok()) {
					std::cerr << "[ttm] ArrowIpcIterator::next error: " << result.status().ToString() << '\n';
					return false;
				}
				out = *result;
				++idx_;
				return true;
			}

			[[nodiscard]] const arrow::Schema& schema() const override {
				return *reader_->schema();
			}

		private:
			std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader_;
			int numBatches_ = 0;
			int idx_        = 0;
		};

		/* =========================================================================
		 * Parquet shard reader
		 * ====================================================================== */

		/**
		 * @brief DatasetIterator for a single Parquet file.
		 */
		class ParquetIterator final : public DatasetIterator {
		public:
			ParquetIterator(
					std::unique_ptr<parquet::arrow::FileReader> reader,
					std::shared_ptr<arrow::Schema> schema
			)
					: reader_(std::move(reader)), schema_(std::move(schema)),
					  numRowGroups_(reader_->num_row_groups()) {}

			bool next(std::shared_ptr<arrow::RecordBatch>& out) override {
				while (idx_ < numRowGroups_) {
					std::shared_ptr<arrow::Table> table;
					const auto status = reader_->ReadRowGroup(idx_, &table);
					++idx_;
					if (!status.ok()) {
						std::cerr << "[ttm] ParquetIterator::next error: " << status.ToString() << '\n';
						continue;
					}
					/* Convert table to a single record batch via TableBatchReader */
					arrow::TableBatchReader tbr(*table);
					const auto batchStatus = tbr.ReadNext(&out);
					if (!batchStatus.ok() || !out) {
						continue;
					}
					return true;
				}
				return false;
			}

			[[nodiscard]] const arrow::Schema& schema() const override { return *schema_; }

		private:
			std::unique_ptr<parquet::arrow::FileReader> reader_;
			std::shared_ptr<arrow::Schema> schema_;
			int numRowGroups_ = 0;
			int idx_          = 0;
		};

		/* =========================================================================
		 * Multi-file (sharded) iterator
		 * ====================================================================== */

		/**
		 * @brief DatasetIterator that chains multiple shard iterators.
		 */
		class MultiFileIterator final : public DatasetIterator {
		public:
			MultiFileIterator(
					std::vector<std::unique_ptr<DatasetIterator>> shards,
					std::shared_ptr<arrow::Schema> schema
			)
					: shards_(std::move(shards)), schema_(std::move(schema)) {}

			bool next(std::shared_ptr<arrow::RecordBatch>& out) override {
				while (shardIdx_ < static_cast<int>(shards_.size())) {
					if (shards_[shardIdx_]->next(out)) {
						return true;
					}
					++shardIdx_;
				}
				return false;
			}

			[[nodiscard]] const arrow::Schema& schema() const override { return *schema_; }

		private:
			std::vector<std::unique_ptr<DatasetIterator>> shards_;
			std::shared_ptr<arrow::Schema> schema_;
			int shardIdx_ = 0;
		};

		/* =========================================================================
		 * Open helpers
		 * ====================================================================== */

		std::expected<std::unique_ptr<DatasetIterator>, std::string>
		open_arrow_ipc(std::unique_ptr<plugins::IByteReader> reader) {
			const int64_t size = probe_size(*reader);
			if (size < 0) {
				return std::unexpected("open_arrow_ipc: reader is not seekable — cannot determine file size");
			}
			auto arrowFile = std::make_shared<IByteReaderFile>(std::move(reader), size);
			auto result    = arrow::ipc::RecordBatchFileReader::Open(arrowFile);
			if (!result.ok()) {
				return std::unexpected("open_arrow_ipc: " + result.status().ToString());
			}
			return std::make_unique<ArrowIpcIterator>(*result);
		}

		std::expected<std::unique_ptr<DatasetIterator>, std::string>
		open_parquet(std::unique_ptr<plugins::IByteReader> reader) {
			const int64_t size = probe_size(*reader);
			if (size < 0) {
				return std::unexpected("open_parquet: reader is not seekable — cannot determine file size");
			}
			auto arrowFile = std::make_shared<IByteReaderFile>(std::move(reader), size);

			std::unique_ptr<parquet::arrow::FileReader> parqReader;
			const auto status = parquet::arrow::OpenFile(
					arrowFile, arrow::default_memory_pool(), &parqReader
			);
			if (!status.ok()) {
				return std::unexpected("open_parquet: " + status.ToString());
			}

			std::shared_ptr<arrow::Schema> schema;
			const auto schemaStatus = parqReader->GetSchema(&schema);
			if (!schemaStatus.ok()) {
				return std::unexpected("open_parquet: GetSchema: " + schemaStatus.ToString());
			}

			return std::make_unique<ParquetIterator>(std::move(parqReader), std::move(schema));
		}

		std::expected<std::unique_ptr<DatasetIterator>, std::string>
		open_shard(plugins::IDatasetSource& source, std::string_view shardUri, const std::string& ext) {
			auto readerOpt = source.open(shardUri);
			if (!readerOpt) {
				return std::unexpected("open_shard: cannot open '" + std::string(shardUri) + "'");
			}
			if (ext == ".arrow") {
				return open_arrow_ipc(std::move(readerOpt));
			}
			if (ext == ".parquet") {
				return open_parquet(std::move(readerOpt));
			}
			return std::unexpected("open_shard: unknown extension '" + ext + "'");
		}

	} // anonymous namespace

	/* =========================================================================
	 * Batch-slicing iterator — wraps any iterator and re-chunks to batch_size
	 * ====================================================================== */

	/**
	 * @brief Wraps a DatasetIterator and slices every returned RecordBatch into
	 *        fixed-size chunks of at most `batch_size` rows.
	 *
	 * Parquet row groups can contain tens of thousands of rows. Without slicing,
	 * the model would receive an enormous batch, causing OOM during the forward
	 * pass. RecordBatch::Slice is zero-copy — it shares the underlying column
	 * buffers — so this adds no data-copy overhead.
	 */
	class BatchSlicingIterator final : public DatasetIterator {
	public:
		BatchSlicingIterator(std::unique_ptr<DatasetIterator> inner, int64_t batch_size)
			: inner_(std::move(inner)), batch_size_(batch_size) {}

		bool next(std::shared_ptr<arrow::RecordBatch>& out) override {
			while (true) {
				// If we have remaining rows in the current chunk, emit a slice
				if (current_ && offset_ < current_->num_rows()) {
					const int64_t take = std::min(current_->num_rows() - offset_, batch_size_);
					out = current_->Slice(offset_, take);
					offset_ += take;
					return true;
				}
				// Fetch the next raw batch from upstream
				if (!inner_->next(current_)) {
					current_.reset();
					return false;
				}
				offset_ = 0;
			}
		}

		[[nodiscard]] const arrow::Schema& schema() const override { return inner_->schema(); }

	private:
		std::unique_ptr<DatasetIterator> inner_;
		int64_t                          batch_size_;
		std::shared_ptr<arrow::RecordBatch> current_;
		int64_t                          offset_ = 0;
	};

	/* =========================================================================
	 * Public API
	 * ====================================================================== */

	std::expected<std::unique_ptr<DatasetIterator>, std::string>
	load_dataset(plugins::IDatasetSource& source, std::string_view uri, std::string_view split, std::string_view config, int64_t batch_size) {
		/* Step 1: Parse the dataset card from README.md */
		/* Open the README.md via the source */
		const std::string readmeUri = std::string(uri) + "/README.md";
		auto readmeReader = source.open(readmeUri);
		if (!readmeReader) {
			/* Try datasetcard.md */
			const std::string cardUri = std::string(uri) + "/datasetcard.md";
			readmeReader = source.open(cardUri);
			if (!readmeReader) {
				return std::unexpected(
						"load_dataset: cannot open README.md or datasetcard.md for '" + std::string(uri) + "'"
				);
			}
		}

		auto& stream = readmeReader->as_stream();
		const std::string readmeContent{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>{}};

		auto infoResult = parse_dataset_card_from_content(readmeContent, config);

		if (!infoResult) {
			return std::unexpected("load_dataset: failed to parse dataset card: " + infoResult.error());
		}
		const DatasetInfo& info = *infoResult;

		/* Step 2: Enumerate split files via the source */
		std::vector<std::string> shardUris;

		/* If the card provides explicit data_files paths, use them directly */
		for (const auto& sp : info.splits) {
			if (sp.name == split && !sp.path.empty()) {
				shardUris.push_back(std::string(uri) + "/" + sp.path);
			}
		}

		if (shardUris.empty()) {
			/* Fallback: probe standard HF sharding pattern data/<split>-NNNNN-of-MMMMM.<ext> */
			for (int shard = 0; shard < 9999; ++shard) {
				const auto shardName = std::format("{:05d}", shard);
				bool found = false;
				for (const char* ext : {".parquet", ".arrow"}) {
					const std::string pattern = std::string(split) + "-" + shardName + ext;
					const std::string shardUri = std::string(uri) + "/data/" + pattern;
					auto probe = source.open(shardUri);
					if (probe) {
						shardUris.push_back(shardUri);
						found = true;
						break;
					}
				}
				if (!found) {
					break;
				}
			}
		}

		if (shardUris.empty()) {
			/* Last resort: single-file split */
			for (const char* ext : {".parquet", ".arrow"}) {
				const std::string singleUri = std::string(uri) + "/data/" + std::string(split) + ext;
				auto probe = source.open(singleUri);
				if (probe) {
					shardUris.push_back(singleUri);
					break;
				}
			}
		}

		if (shardUris.empty()) {
			return std::unexpected(
					"load_dataset: no data files found for split '" + std::string(split) + "' in '" +
					std::string(uri) + "'"
			);
		}

		/* Step 3: Open each shard */
		std::vector<std::unique_ptr<DatasetIterator>> shards;
		std::shared_ptr<arrow::Schema> schema;

		for (const auto& shardUri : shardUris) {
			const std::string ext = std::filesystem::path(shardUri).extension().string();
			auto shardResult = open_shard(source, shardUri, ext);
			if (!shardResult) {
				std::cerr << "[ttm] load_dataset: skipping shard '" << shardUri
						  << "': " << shardResult.error() << '\n';
				continue;
			}
			if (!schema) {
				schema = std::make_shared<arrow::Schema>((*shardResult)->schema());
			}
			shards.push_back(std::move(*shardResult));
		}

		if (shards.empty()) {
			return std::unexpected("load_dataset: no shards could be opened for split '" + std::string(split) + "'");
		}

		std::unique_ptr<DatasetIterator> result =
			std::make_unique<MultiFileIterator>(std::move(shards), std::move(schema));

		if (batch_size > 0) {
			result = std::make_unique<BatchSlicingIterator>(std::move(result), batch_size);
		}

		return result;
	}

} // namespace ttm::datasets

/**
 * @file dataset_loader.hpp
 * @brief High-level dataset loading interface.
 *
 * @details
 * Combines the plugin-provided IDatasetSource (for URI-based file access) with
 * the dataset metadata parser to produce an iterable stream of Apache Arrow
 * RecordBatches.
 *
 * Supported file formats (auto-detected from file extension):
 * - `.arrow` — Arrow IPC stream / random-access file format
 * - `.parquet` — Apache Parquet columnar format
 *
 * @par Typical usage
 * @code{.cpp}
 * auto& source = *manager.find_source("hf:");
 * auto result  = ttm::datasets::load_dataset(source, "hf:ylecun/mnist", "train");
 * auto iter    = std::move(result).value();
 *
 * std::shared_ptr<arrow::RecordBatch> batch;
 * while (iter->next(batch)) {
 *     // process batch ...
 * }
 * @endcode
 */

#ifndef TTM_DATASETS_DATASET_LOADER_HPP
#define TTM_DATASETS_DATASET_LOADER_HPP

#include <ttm/compat/expected.hpp>
#include <ttm/datasets/dataset_info.hpp>
#include <ttm/plugins/extension.hpp>

#include <memory>
#include <string>
#include <string_view>

/* Forward declarations for Apache Arrow types */
namespace arrow {
	class RecordBatch;
	class Schema;
} // namespace arrow

namespace ttm::datasets {

	/**
	 * @brief Abstract iterator over a stream of Arrow RecordBatches.
	 *
	 * @details
	 * Yields one RecordBatch per call to next().  All batches share the same
	 * schema, accessible via schema().
	 *
	 * Implementations may read from a single Arrow IPC file, a Parquet file,
	 * or a concatenation of multiple files (e.g. sharded HuggingFace datasets).
	 */
	class DatasetIterator {
	public:
		DatasetIterator()          = default;
		virtual ~DatasetIterator() = default;

		DatasetIterator(const DatasetIterator&)            = delete;
		DatasetIterator& operator=(const DatasetIterator&) = delete;
		DatasetIterator(DatasetIterator&&)                 = delete;
		DatasetIterator& operator=(DatasetIterator&&)      = delete;

		/**
		 * @brief Advance to the next batch.
		 *
		 * @param[out] out  Set to the next RecordBatch on success.
		 * @return `true` if a batch was produced, `false` at end-of-stream.
		 */
		virtual bool next(std::shared_ptr<arrow::RecordBatch>& out) = 0;

		/**
		 * @brief Return the shared schema of all batches produced by this iterator.
		 */
		[[nodiscard]] virtual const arrow::Schema& schema() const = 0;
	};

	/**
	 * @brief Open a dataset and return an iterator over its record batches.
	 *
	 * @details
	 * Loading sequence:
	 * 1. Opens `<uri>/README.md` (or `datasetcard.md`) via the source.
	 * 2. Parses the HuggingFace Dataset Card frontmatter, selecting `config` if non-empty.
	 * 3. If the selected config provides explicit `data_files` paths, uses those directly.
	 *    Otherwise falls back to probing `<uri>/data/<split>-*.parquet` / `*.arrow`.
	 * 4. Opens each data file via the source and wraps it in an Arrow reader.
	 * 5. Returns a DatasetIterator that chains all shard readers.
	 *
	 * @param[in] source   IDatasetSource that handles the URI scheme (e.g. "hf:").
	 * @param[in] uri      Dataset URI (e.g. "hf:thagen/SCITE").
	 * @param[in] split    Split name (default: "train").
	 * @param[in] config   Config name (e.g. "causality detection"); empty → first / only config.
	 *
	 * @return A DatasetIterator on success, or an error string on failure.
	 */
	[[nodiscard]] std::expected<std::unique_ptr<DatasetIterator>, std::string> load_dataset(
			ttm::plugins::IDatasetSource& source,
			std::string_view uri,
			std::string_view split  = "train",
			std::string_view config = ""
	);

} // namespace ttm::datasets

#endif /* TTM_DATASETS_DATASET_LOADER_HPP */

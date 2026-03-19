/**
 * @brief Demo: load the SCITE "causality detection" subset from HuggingFace.
 *
 * Dataset: https://huggingface.co/datasets/thagen/SCITE
 * Subset:  causality detection
 *
 * Build with extensions enabled:
 *   cmake -B build -DTTM_BUILD_EXTENSIONS=ON
 *   cmake --build build
 *
 * The hf: URI scheme is handled by extensions/core (ttm_core.so / .dylib / .dll).
 * That plugin clones the dataset repo via libgit2 into a local cache
 * ($XDG_CACHE_HOME/ttm/datasets/<hash>/) and serves files from there.
 *
 * HuggingFace Parquet layout for a named subset:
 *   data/<config_name>/<split>-NNNNN-of-MMMMM.parquet
 *
 * So the URI we pass to load_dataset() includes the config path:
 *   hf:thagen/SCITE/causality detection
 * which causes the loader to probe:
 *   data/train-00000-of-MMMMM.parquet  (inside that subtree)
 */

#include <ttm/datasets/dataset_info.hpp>
#include <ttm/datasets/dataset_loader.hpp>
#include <ttm/plugins/plugin_manager.hpp>

#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/scalar.h>
#include <arrow/type.h>

#include <filesystem>
#include <ttm/compat/format.hpp>
#include <iostream>
#include <string_view>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

	/// Print the Arrow schema (column name + type) to stdout.
	void print_schema(const arrow::Schema& schema) {
		std::cout << "Schema (" << schema.num_fields() << " columns):\n";
		for (int i = 0; i < schema.num_fields(); ++i) {
			const auto& field = *schema.field(i);
			std::cout << std::format("  [{:2d}] {:30s}  {}\n", i, field.name(), field.type()->ToString());
		}
		std::cout << '\n';
	}

	/// Print up to `max_rows` rows from a single RecordBatch as a simple table.
	void print_batch(const arrow::RecordBatch& batch, int max_rows = 5) {
		const int rows = static_cast<int>(std::min(static_cast<int64_t>(max_rows), batch.num_rows()));
		std::cout << std::format(
				"  Batch: {} rows × {} columns  (showing first {})\n", batch.num_rows(), batch.num_columns(), rows
		);

		for (int r = 0; r < rows; ++r) {
			std::cout << "  row " << r << ":";
			for (int c = 0; c < batch.num_columns(); ++c) {
				const auto& col = *batch.column(c);
				const auto& name = batch.schema()->field(c)->name();
				// Use Arrow's generic ToString (available on all array types)
				std::cout << std::format("  {}={}", name, col.GetScalar(r).ValueOrDie()->ToString());
			}
			std::cout << '\n';
		}
		std::cout << '\n';
	}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int /*argc*/, char* /*argv*/[]) {
	// ── 1. Initialise the plugin manager ──────────────────────────────────
	auto mgrResult = ttm::plugins::PluginManager::create();
	if (!mgrResult) {
		std::cerr << "Failed to create PluginManager: " << mgrResult.error() << '\n';
		return 1;
	}
	auto& mgr = *mgrResult;

	// ── 2. Load the core native plugin (provides hf:, gh:, gl:, bb:, sr:) ─
	//
	// The plugin is built to build/extensions/core/ttm_core.so when TTM_BUILD_EXTENSIONS=ON.
	// Adjust the path if your build directory differs.
	const std::filesystem::path corePath = std::filesystem::path(__FILE__)
												   .parent_path() // src/
												   .parent_path() // project root
										   / "build" / "extensions" / "core" / "ttm_core.so";

	if (auto r = mgr.load(corePath); !r) {
		std::cerr << "Failed to load core plugin (" << corePath.string() << "): " << r.error() << '\n';
		std::cerr << "Build with: cmake -B build -DTTM_BUILD_EXTENSIONS=ON && cmake --build build\n";
		return 1;
	}
	std::cout << "Core plugin loaded.\n";

	// ── 3. Resolve the hf: dataset source ─────────────────────────────────
	auto* hfSource = mgr.find_source("hf:");
	if (hfSource == nullptr) {
		std::cerr << "hf: source not registered — is the core plugin loaded?\n";
		return 1;
	}

	// ── 4. Load the dataset ────────────────────────────────────────────────
	//
	// URI format:  hf:<owner>/<repo>[@<ref>][/<subpath>]
	//
	// The SCITE "causality detection" config stores its Parquet files at:
	//   data/causality detection/<split>-NNNNN-of-MMMMM.parquet
	//
	// We encode the config name as the subpath so load_dataset() probes the
	// right directory.  Spaces in config names are fine; the core plugin passes
	// the full path to libgit2's file access.
	constexpr std::string_view kDatasetUri = "hf:thagen/SCITE";
	constexpr std::string_view kConfig     = "causality detection";
	constexpr std::string_view kSplit      = "train";

	std::cout << std::format("Loading dataset  : {}\n", kDatasetUri);
	std::cout << std::format("Config           : {}\n", kConfig);
	std::cout << std::format("Split            : {}\n\n", kSplit);

	auto iterResult = ttm::datasets::load_dataset(*hfSource, kDatasetUri, kSplit, kConfig);
	if (!iterResult) {
		std::cerr << "Failed to load dataset: " << iterResult.error() << '\n';
		return 1;
	}
	auto& iter = *iterResult;

	// ── 5. Print schema ────────────────────────────────────────────────────
	print_schema(iter->schema());

	// ── 6. Iterate batches ─────────────────────────────────────────────────
	std::shared_ptr<arrow::RecordBatch> batch;
	int64_t totalRows = 0;
	int batchCount = 0;

	while (iter->next(batch)) {
		++batchCount;
		totalRows += batch->num_rows();

		if (batchCount <= 2) {
			std::cout << std::format("── Batch {} ────────────────────────────────\n", batchCount);
			print_batch(*batch, /*max_rows=*/5);
		}
	}

	std::cout << std::format("Done.  {} batches, {} total rows.\n", batchCount, totalRows);
	return 0;
}

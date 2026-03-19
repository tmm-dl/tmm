/**
 * @file dataset_info.hpp
 * @brief HuggingFace Dataset Card metadata types and parser.
 *
 * @details
 * Parses the YAML frontmatter embedded in a HuggingFace Dataset Card
 * (`README.md` or `datasetcard.md`).  The expected structure follows the
 * official HF hub-docs schema:
 *
 * ```yaml
 * ---
 * dataset_info:
 *   features:
 *     - name: text
 *       dtype: string
 *     - name: label
 *       dtype:
 *         class_label:
 *           names:
 *             - positive
 *             - negative
 *   splits:
 *     - name: train
 *       num_examples: 5000
 *     - name: test
 *       num_examples: 1000
 * ---
 * ```
 *
 * @see parse_dataset_card   Read and parse a card from a local repo root
 * @see find_split_files     Enumerate split data files
 */

#ifndef TTM_DATASETS_DATASET_INFO_HPP
#define TTM_DATASETS_DATASET_INFO_HPP

#include <ttm/compat/expected.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ttm::datasets {

	/* =========================================================================
	 * Feature types
	 * ====================================================================== */

	/**
	 * @brief Semantic kind of a dataset feature column.
	 */
	enum class FeatureKind : uint8_t {
		Scalar,     ///< Simple scalar value (string, int, float, …).
		Sequence,   ///< Repeated nested feature (list/array).
		ClassLabel, ///< Categorical label with a fixed set of class names.
		Image,      ///< Raw image bytes.
		Audio,      ///< Raw audio data.
		Unknown,    ///< Unrecognised or complex dtype.
	};

	/**
	 * @brief Metadata for a single feature column in the dataset schema.
	 */
	struct DatasetFeature {
		std::string name;    ///< Column name as declared in the card.
		std::string dtype;   ///< Raw dtype string (e.g. "string", "int64", "float32").
		FeatureKind kind = FeatureKind::Unknown;

		/** Class names for ClassLabel features; empty for other kinds. */
		std::vector<std::string> class_names;

		/** Nested sub-features for Sequence features; empty for other kinds. */
		std::vector<DatasetFeature> sequence_feature;
	};

	/* =========================================================================
	 * Split types
	 * ====================================================================== */

	/**
	 * @brief Metadata for one data split (train / validation / test / …).
	 */
	struct DatasetSplit {
		std::string name;                 ///< Split name, e.g. "train".
		std::string path;                 ///< Explicit file path relative to repo root (from configs.data_files); empty if not specified.
		int64_t     num_examples = -1;    ///< Row count, or -1 if unknown.
		int64_t     num_bytes    = -1;    ///< Uncompressed size in bytes, or -1 if unknown.
	};

	/* =========================================================================
	 * DatasetInfo — top-level card metadata
	 * ====================================================================== */

	/**
	 * @brief Aggregated metadata extracted from a HuggingFace Dataset Card.
	 */
	struct DatasetInfo {
		std::string pretty_name;   ///< Human-readable dataset name (optional).
		std::string config_name;   ///< Configuration name (e.g. "default", "en").

		std::vector<DatasetFeature> features; ///< Schema columns.
		std::vector<DatasetSplit>   splits;   ///< Available data splits.

		/** Task categories declared in the card (e.g. "text-classification"). */
		std::vector<std::string> task_categories;
	};

	/* =========================================================================
	 * Parser and file helpers
	 * ====================================================================== */

	/**
	 * @brief Parse a HuggingFace Dataset Card from a local repository root.
	 *
	 * @details
	 * Looks for `README.md` (preferred) or `datasetcard.md` in `repo_root`.
	 * Extracts the YAML frontmatter block (`---\n…\n---`) and handles two formats:
	 *
	 * - `configs:` list (modern HF format): each entry has `config_name`,
	 *   `data_files` (explicit split → path mappings), and `features`.
	 * - `dataset_info:` mapping or list (legacy HF format).
	 *
	 * When `config_name` is non-empty the matching config is selected;
	 * otherwise the first available config is used.
	 *
	 * @param[in] repo_root    Path to the local clone of the dataset repository.
	 * @param[in] config_name  Config to select (e.g. "causality detection"); empty → first config.
	 * @return Parsed DatasetInfo on success, or an error string on failure.
	 */
	[[nodiscard]] std::expected<DatasetInfo, std::string>
	parse_dataset_card(const std::filesystem::path& repo_root, std::string_view config_name = "");

	/**
	 * @brief Parse a HuggingFace Dataset Card directly from its Markdown content.
	 *
	 * @details
	 * Same as `parse_dataset_card` but operates on an in-memory string rather
	 * than a file, avoiding a disk round-trip when the card was already fetched
	 * through a plugin source.
	 *
	 * @param[in] markdown     Raw Markdown text (must begin with `---` frontmatter).
	 * @param[in] config_name  Config to select; empty → first config.
	 */
	[[nodiscard]] std::expected<DatasetInfo, std::string>
	parse_dataset_card_from_content(std::string_view markdown, std::string_view config_name = "");

	/**
	 * @brief Enumerate the data files for a given split.
	 *
	 * @details
	 * Scans the `data/` subdirectory of `repo_root` for files whose basename
	 * starts with `<split>-` and ends with `.parquet` or `.arrow`.
	 * Files are returned in lexicographic order.
	 *
	 * @param[in] info       Parsed dataset metadata (used for split validation).
	 * @param[in] split      Split name (e.g. "train", "test").
	 * @param[in] repo_root  Path to the local repository root.
	 * @return Vector of absolute file paths for the split; empty if none found.
	 */
	[[nodiscard]] std::vector<std::filesystem::path>
	find_split_files(const DatasetInfo& info, std::string_view split, const std::filesystem::path& repo_root);

} // namespace ttm::datasets

#endif /* TTM_DATASETS_DATASET_INFO_HPP */

/**
 * @file dataset_info.cpp
 * @brief HuggingFace Dataset Card metadata parser.
 *
 * @details
 * Parses YAML frontmatter (`---\n…\n---`) from a HuggingFace Dataset Card
 * (README.md or datasetcard.md) using yaml-cpp.  Handles:
 * - Simple scalar dtype columns (string, int32, float64, …)
 * - ClassLabel features (dtype: {class_label: {names: […] or {'0': …}}})
 * - Sequence features (dtype: {sequence: …})
 * - Image / Audio feature markers
 * - task_categories list
 * - Modern `configs:` format (explicit data_files per split)
 * - Legacy `dataset_info:` format (single mapping or list of configs)
 */

#include <ttm/datasets/dataset_info.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <ttm/compat/expected.hpp>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace ttm::datasets {

	namespace {

		/* =========================================================================
		 * Frontmatter extraction
		 * ====================================================================== */

		/**
		 * @brief Extract YAML frontmatter (`---\n…\n---`) from any istream.
		 * @param source  Human-readable name used in error messages (file path or "<inline>").
		 */
		std::expected<std::string, std::string> extract_frontmatter(std::istream& is, std::string_view source) {
			std::string line;
			/* Skip leading blank lines; first non-blank line must be "---" */
			while (std::getline(is, line)) {
				if (!line.empty() && line != "\r")
					break;
			}
			if (!line.empty() && line.back() == '\r')
				line.pop_back();
			if (line != "---") {
				return std::unexpected(
						"parse_dataset_card: '" + std::string(source) + "' does not start with YAML front matter (---)"
				);
			}

			std::ostringstream yaml;
			while (std::getline(is, line)) {
				if (!line.empty() && line.back() == '\r')
					line.pop_back();
				if (line == "---")
					break;
				yaml << line << '\n';
			}
			return yaml.str();
		}

		/* =========================================================================
		 * Feature parsing helpers
		 * ====================================================================== */

		// Forward declarations (parse_feature_from_dtype ↔ make_sequence_inner are mutually recursive)
		DatasetFeature parse_feature(const YAML::Node& node);
		DatasetFeature make_sequence_inner(const std::string& name, const YAML::Node& seqVal);

		/**
		 * @brief Parse a class_label `names` node that may be a sequence or an
		 *        integer-keyed map (e.g. `'0': uncausal, '1': causal`).
		 */
		std::vector<std::string> parse_class_names(const YAML::Node& namesNode) {
			std::vector<std::string> names;
			if (namesNode.IsSequence()) {
				for (const auto& n : namesNode) {
					names.push_back(n.as<std::string>());
				}
			} else if (namesNode.IsMap()) {
				std::vector<std::pair<int, std::string>> entries;
				for (const auto& kv : namesNode) {
					entries.emplace_back(std::stoi(kv.first.as<std::string>()), kv.second.as<std::string>());
				}
				std::sort(entries.begin(), entries.end());
				for (auto& [k, v] : entries) {
					names.push_back(std::move(v));
				}
			}
			return names;
		}

		// NOLINTNEXTLINE(misc-no-recursion) -- intentional: sequence features may be nested; depth in practice is ≤2
		DatasetFeature parse_feature_from_dtype(const std::string& name, const YAML::Node& dtypeNode) {
			DatasetFeature feat;
			feat.name = name;

			if (!dtypeNode) {
				feat.kind = FeatureKind::Unknown;
				return feat;
			}

			if (dtypeNode.IsScalar()) {
				feat.dtype = dtypeNode.as<std::string>();
				if (feat.dtype == "image") {
					feat.kind = FeatureKind::Image;
				} else if (feat.dtype == "audio") {
					feat.kind = FeatureKind::Audio;
				} else {
					feat.kind = FeatureKind::Scalar;
				}
				return feat;
			}

			if (dtypeNode.IsMap()) {
				if (const YAML::Node cl = dtypeNode["class_label"]) {
					feat.kind = FeatureKind::ClassLabel;
					feat.dtype = "class_label";
					if (cl["names"])
						feat.class_names = parse_class_names(cl["names"]);
					return feat;
				}
				if (dtypeNode["sequence"]) {
					feat.kind = FeatureKind::Sequence;
					feat.dtype = "sequence";
					feat.sequence_feature.push_back(make_sequence_inner(name, dtypeNode["sequence"]));
					return feat;
				}
				if (dtypeNode["image"]) {
					feat.kind = FeatureKind::Image;
					feat.dtype = "image";
					return feat;
				}
				if (dtypeNode["audio"]) {
					feat.kind = FeatureKind::Audio;
					feat.dtype = "audio";
					return feat;
				}
			}

			feat.kind = FeatureKind::Unknown;
			return feat;
		}

		/**
		 * @brief Build the inner DatasetFeature for a sequence element.
		 *
		 * `seqVal` is the value of the `sequence:` key — either a scalar type
		 * string or a nested dtype map.
		 */
		DatasetFeature make_sequence_inner(const std::string& name, const YAML::Node& seqVal) {
			DatasetFeature inner;
			inner.name = name + "_item";
			if (seqVal.IsScalar()) {
				inner.dtype = seqVal.as<std::string>();
				inner.kind = FeatureKind::Scalar;
			} else {
				inner = parse_feature_from_dtype(name + "_item", seqVal);
			}
			return inner;
		}

		DatasetFeature parse_feature(const YAML::Node& node) {
			DatasetFeature feat;
			if (!node.IsMap())
				return feat;
			if (node["name"])
				feat.name = node["name"].as<std::string>();

			if (node["dtype"]) {
				feat = parse_feature_from_dtype(feat.name, node["dtype"]);
			} else if (node["sequence"]) {
				feat.kind = FeatureKind::Sequence;
				feat.dtype = "sequence";
				feat.sequence_feature.push_back(make_sequence_inner(feat.name, node["sequence"]));
			} else if (const YAML::Node cl = node["class_label"]) {
				feat.kind = FeatureKind::ClassLabel;
				feat.dtype = "class_label";
				if (cl["names"])
					feat.class_names = parse_class_names(cl["names"]);
			} else if (node["image"]) {
				feat.kind = FeatureKind::Image;
				feat.dtype = "image";
			} else if (node["audio"]) {
				feat.kind = FeatureKind::Audio;
				feat.dtype = "audio";
			}

			return feat;
		}

		/* =========================================================================
		 * Legacy dataset_info block parsing
		 * ====================================================================== */

		DatasetInfo parse_dataset_info_node(const YAML::Node& infoNode) {
			DatasetInfo info;
			if (infoNode["config_name"])
				info.config_name = infoNode["config_name"].as<std::string>();
			if (infoNode["features"] && infoNode["features"].IsSequence()) {
				for (const auto& f : infoNode["features"])
					info.features.push_back(parse_feature(f));
			}
			if (infoNode["splits"] && infoNode["splits"].IsSequence()) {
				for (const auto& s : infoNode["splits"]) {
					DatasetSplit split;
					if (s["name"])
						split.name = s["name"].as<std::string>();
					if (s["num_examples"])
						split.num_examples = s["num_examples"].as<int64_t>();
					if (s["num_bytes"])
						split.num_bytes = s["num_bytes"].as<int64_t>();
					info.splits.push_back(std::move(split));
				}
			}
			return info;
		}

		/* =========================================================================
		 * YAML string → DatasetInfo
		 * ====================================================================== */

		std::expected<DatasetInfo, std::string>
		parse_yaml_into_info(const std::string& yaml_str, std::string_view config_name) {
			YAML::Node root;
			try {
				root = YAML::Load(yaml_str);
			} catch (const YAML::Exception& e) {
				return std::unexpected(std::string("parse_dataset_card: YAML parse error: ") + e.what());
			}

			DatasetInfo result;
			if (root["pretty_name"])
				result.pretty_name = root["pretty_name"].as<std::string>();
			if (root["task_categories"] && root["task_categories"].IsSequence()) {
				for (const auto& tc : root["task_categories"]) {
					result.task_categories.push_back(tc.as<std::string>());
				}
			}

			/* Modern HF format: configs list with explicit data_files */
			if (root["configs"] && root["configs"].IsSequence()) {
				const auto& configs = root["configs"];
				int chosenIdx = -1;
				for (std::size_t i = 0; i < configs.size(); ++i) {
					const YAML::Node cfg = configs[i];
					if (config_name.empty() ||
						(cfg["config_name"] && cfg["config_name"].as<std::string>() == config_name)) {
						chosenIdx = static_cast<int>(i);
						break;
					}
				}
				if (chosenIdx < 0) {
					return std::unexpected("parse_dataset_card: config '" + std::string(config_name) + "' not found");
				}
				const YAML::Node cfg = configs[chosenIdx];
				if (cfg["config_name"])
					result.config_name = cfg["config_name"].as<std::string>();
				if (cfg["features"] && cfg["features"].IsSequence()) {
					for (const auto& f : cfg["features"])
						result.features.push_back(parse_feature(f));
				}
				if (cfg["data_files"] && cfg["data_files"].IsSequence()) {
					for (const auto& df : cfg["data_files"]) {
						DatasetSplit sp;
						if (df["split"])
							sp.name = df["split"].as<std::string>();
						if (df["path"])
							sp.path = df["path"].as<std::string>();
						result.splits.push_back(std::move(sp));
					}
				}
				return result;
			}

			/* Legacy HF format: dataset_info mapping or list */
			if (root["dataset_info"]) {
				const auto& di = root["dataset_info"];
				if (di.IsMap()) {
					auto info = parse_dataset_info_node(di);
					result.features = std::move(info.features);
					result.splits = std::move(info.splits);
					if (result.config_name.empty())
						result.config_name = info.config_name;
				} else if (di.IsSequence() && di.size() > 0) {
					int chosenIdx = -1;
					for (std::size_t i = 0; i < di.size(); ++i) {
						const YAML::Node entry = di[i];
						if (config_name.empty() ||
							(entry["config_name"] && entry["config_name"].as<std::string>() == config_name)) {
							chosenIdx = static_cast<int>(i);
							break;
						}
					}
					if (chosenIdx >= 0) {
						const YAML::Node entry = di[chosenIdx];
						auto info = parse_dataset_info_node(entry);
						result.features = std::move(info.features);
						result.splits = std::move(info.splits);
						result.config_name = info.config_name;
					}
				}
			}

			return result;
		}

	} // anonymous namespace

	/* =========================================================================
	 * Public API
	 * ====================================================================== */

	std::expected<DatasetInfo, std::string>
	parse_dataset_card(const std::filesystem::path& repo_root, std::string_view config_name) {
		const std::filesystem::path readmePath = repo_root / "README.md";
		const std::filesystem::path cardPath = repo_root / "datasetcard.md";

		const std::filesystem::path* cardFile = std::filesystem::exists(readmePath) ? &readmePath
												: std::filesystem::exists(cardPath) ? &cardPath
																					: nullptr;
		if (cardFile == nullptr) {
			return std::unexpected(
					"parse_dataset_card: no README.md or datasetcard.md in '" + repo_root.string() + "'"
			);
		}

		std::ifstream ifs(*cardFile);
		if (!ifs) {
			return std::unexpected("parse_dataset_card: cannot open '" + cardFile->string() + "'");
		}

		auto yamlStr = extract_frontmatter(ifs, cardFile->string());
		if (!yamlStr)
			return std::unexpected(yamlStr.error());
		return parse_yaml_into_info(*yamlStr, config_name);
	}

	std::expected<DatasetInfo, std::string>
	parse_dataset_card_from_content(std::string_view markdown, std::string_view config_name) {
		std::istringstream iss{std::string(markdown)};
		auto yamlStr = extract_frontmatter(iss, "<inline content>");
		if (!yamlStr)
			return std::unexpected(yamlStr.error());
		return parse_yaml_into_info(*yamlStr, config_name);
	}

	std::vector<std::filesystem::path>
	find_split_files(const DatasetInfo& /*info*/, std::string_view split, const std::filesystem::path& repo_root) {
		const std::filesystem::path dataDir = repo_root / "data";
		if (!std::filesystem::is_directory(dataDir))
			return {};

		const std::string prefix = std::string(split) + "-";
		std::vector<std::filesystem::path> result;

		for (const auto& entry : std::filesystem::directory_iterator(dataDir)) {
			if (!entry.is_regular_file())
				continue;
			const auto& p = entry.path();
			const auto ext = p.extension().string();
			if (ext != ".parquet" && ext != ".arrow")
				continue;
			if (p.filename().string().starts_with(prefix))
				result.push_back(p);
		}

		std::sort(result.begin(), result.end());
		return result;
	}

} // namespace ttm::datasets

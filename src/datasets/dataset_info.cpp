/**
 * @file dataset_info.cpp
 * @brief HuggingFace Dataset Card metadata parser.
 *
 * @details
 * Parses YAML frontmatter (`---\n…\n---`) from a HuggingFace Dataset Card
 * (README.md or datasetcard.md) using yaml-cpp.  Handles:
 * - Simple scalar dtype columns (string, int32, float64, …)
 * - ClassLabel features (dtype: {class_label: {names: […]}})
 * - Sequence features (dtype: {sequence: …})
 * - Image / Audio feature markers
 * - task_categories list
 * - Multiple configs (dataset_info is a list): uses first entry
 */

#include <ttm/datasets/dataset_info.hpp>

#include <algorithm>
#include <cstdlib>
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
		 * YAML frontmatter extraction
		 * ====================================================================== */

		/**
		 * @brief Extract the YAML frontmatter block from a Markdown file.
		 *
		 * @details
		 * Looks for the first `---` delimiter on line 1, then reads until the
		 * next `---` delimiter.  Returns the content between the delimiters.
		 */
		std::expected<std::string, std::string> extract_frontmatter(const std::filesystem::path& path) {
			std::ifstream ifs(path);
			if (!ifs) {
				return std::unexpected("extract_frontmatter: cannot open '" + path.string() + "'");
			}

			std::string line;
			/* First non-empty line must be "---" */
			while (std::getline(ifs, line)) {
				if (!line.empty() && line != "\r") {
					break;
				}
			}
			/* Trim trailing CR if any */
			if (!line.empty() && line.back() == '\r') {
				line.pop_back();
			}
			if (line != "---") {
				return std::unexpected(
						"extract_frontmatter: '" + path.string() + "' does not start with YAML front matter (---)"
				);
			}

			std::ostringstream yaml;
			while (std::getline(ifs, line)) {
				if (!line.empty() && line.back() == '\r') {
					line.pop_back();
				}
				if (line == "---") {
					break;
				}
				yaml << line << '\n';
			}
			return yaml.str();
		}

		/* =========================================================================
		 * Feature parsing helpers
		 * ====================================================================== */

		/**
		 * @brief Parse a dtype node which may be a string or a mapping.
		 *
		 * Handles:
		 * - string → Scalar
		 * - {class_label: {names: [...]}} → ClassLabel
		 * - {sequence: <inner>} → Sequence
		 * - {image: ...} → Image
		 * - {audio: ...} → Audio
		 */
		// NOLINTNEXTLINE(misc-no-recursion) -- intentional: Sequence features may be nested; depth in practice is ≤2
		DatasetFeature parse_feature(const YAML::Node& node);

		DatasetFeature parse_feature_from_dtype(const std::string& name, const YAML::Node& dtypeNode) {
			DatasetFeature feat;
			feat.name = name;

			if (!dtypeNode) {
				feat.kind = FeatureKind::Unknown;
				return feat;
			}

			if (dtypeNode.IsScalar()) {
				feat.dtype = dtypeNode.as<std::string>();
				const auto& dt = feat.dtype;
				if (dt == "image") {
					feat.kind = FeatureKind::Image;
				} else if (dt == "audio") {
					feat.kind = FeatureKind::Audio;
				} else {
					feat.kind = FeatureKind::Scalar;
				}
				return feat;
			}

			if (dtypeNode.IsMap()) {
				if (dtypeNode["class_label"]) {
					feat.kind = FeatureKind::ClassLabel;
					feat.dtype = "class_label";
					const auto& clNode = dtypeNode["class_label"];
					if (clNode["names"]) {
						const auto& namesNode = clNode["names"];
						if (namesNode.IsSequence()) {
							for (const auto& n : namesNode) {
								feat.class_names.push_back(n.as<std::string>());
							}
						} else if (namesNode.IsMap()) {
							/* Map form: '0': uncausal, '1': causal — sort by integer key */
							std::vector<std::pair<int, std::string>> entries;
							for (const auto& kv : namesNode) {
								entries.emplace_back(std::stoi(kv.first.as<std::string>()), kv.second.as<std::string>());
							}
							std::sort(entries.begin(), entries.end());
							for (auto& [k, v] : entries) {
								feat.class_names.push_back(std::move(v));
							}
						}
					}
					return feat;
				}
				if (dtypeNode["sequence"]) {
					feat.kind = FeatureKind::Sequence;
					feat.dtype = "sequence";
					/* sequence value may be a string (simple element type) or a mapping */
					const auto& seqVal = dtypeNode["sequence"];
					DatasetFeature inner;
					inner.name = name + "_item";
					if (seqVal.IsScalar()) {
						inner.dtype = seqVal.as<std::string>();
						inner.kind = FeatureKind::Scalar;
					} else {
						inner = parse_feature_from_dtype(name + "_item", seqVal);
					}
					feat.sequence_feature.push_back(std::move(inner));
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

		DatasetFeature parse_feature(const YAML::Node& node) {
			DatasetFeature feat;
			if (!node.IsMap()) {
				return feat;
			}

			if (node["name"]) {
				feat.name = node["name"].as<std::string>();
			}

			/* dtype can be omitted for nested features */
			if (node["dtype"]) {
				feat = parse_feature_from_dtype(feat.name, node["dtype"]);
			} else if (node["sequence"]) {
				/* Short form: {name: ..., sequence: <inner>} */
				feat.kind = FeatureKind::Sequence;
				feat.dtype = "sequence";
				const auto& seqVal = node["sequence"];
				DatasetFeature inner;
				inner.name = feat.name + "_item";
				if (seqVal.IsScalar()) {
					inner.dtype = seqVal.as<std::string>();
					inner.kind = FeatureKind::Scalar;
				} else {
					inner = parse_feature_from_dtype(feat.name + "_item", seqVal);
				}
				feat.sequence_feature.push_back(std::move(inner));
			} else if (node["class_label"]) {
				feat.kind = FeatureKind::ClassLabel;
				feat.dtype = "class_label";
				const auto& clNode = node["class_label"];
				if (clNode["names"]) {
					const auto& namesNode = clNode["names"];
					if (namesNode.IsSequence()) {
						for (const auto& n : namesNode) {
							feat.class_names.push_back(n.as<std::string>());
						}
					} else if (namesNode.IsMap()) {
						std::vector<std::pair<int, std::string>> entries;
						for (const auto& kv : namesNode) {
							entries.emplace_back(std::stoi(kv.first.as<std::string>()), kv.second.as<std::string>());
						}
						std::sort(entries.begin(), entries.end());
						for (auto& [k, v] : entries) {
							feat.class_names.push_back(std::move(v));
						}
					}
				}
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
		 * Split parsing
		 * ====================================================================== */

		DatasetSplit parse_split(const YAML::Node& node) {
			DatasetSplit split;
			if (node["name"]) {
				split.name = node["name"].as<std::string>();
			}
			if (node["num_examples"]) {
				split.num_examples = node["num_examples"].as<int64_t>();
			}
			if (node["num_bytes"]) {
				split.num_bytes = node["num_bytes"].as<int64_t>();
			}
			return split;
		}

		/* =========================================================================
		 * dataset_info block parsing
		 * ====================================================================== */

		DatasetInfo parse_dataset_info_node(const YAML::Node& infoNode) {
			DatasetInfo info;

			if (infoNode["config_name"]) {
				info.config_name = infoNode["config_name"].as<std::string>();
			}

			if (infoNode["features"] && infoNode["features"].IsSequence()) {
				for (const auto& f : infoNode["features"]) {
					info.features.push_back(parse_feature(f));
				}
			}

			if (infoNode["splits"] && infoNode["splits"].IsSequence()) {
				for (const auto& s : infoNode["splits"]) {
					info.splits.push_back(parse_split(s));
				}
			}

			return info;
		}

	} // anonymous namespace

	/* =========================================================================
	 * Public API
	 * ====================================================================== */

	std::expected<DatasetInfo, std::string>
	parse_dataset_card(const std::filesystem::path& repo_root, std::string_view config_name) {
		/* Find the card file */
		const std::filesystem::path readmePath = repo_root / "README.md";
		const std::filesystem::path cardPath   = repo_root / "datasetcard.md";

		std::filesystem::path cardFile;
		if (std::filesystem::exists(readmePath)) {
			cardFile = readmePath;
		} else if (std::filesystem::exists(cardPath)) {
			cardFile = cardPath;
		} else {
			return std::unexpected(
					"parse_dataset_card: no README.md or datasetcard.md in '" + repo_root.string() + "'"
			);
		}

		/* Extract YAML frontmatter */
		auto yamlStr = extract_frontmatter(cardFile);
		if (!yamlStr) {
			return std::unexpected(yamlStr.error());
		}

		/* Parse YAML */
		YAML::Node root;
		try {
			root = YAML::Load(*yamlStr);
		} catch (const YAML::Exception& e) {
			return std::unexpected(
					std::string("parse_dataset_card: YAML parse error: ") + e.what()
			);
		}

		DatasetInfo result;

		/* pretty_name */
		if (root["pretty_name"]) {
			result.pretty_name = root["pretty_name"].as<std::string>();
		}

		/* task_categories */
		if (root["task_categories"] && root["task_categories"].IsSequence()) {
			for (const auto& tc : root["task_categories"]) {
				result.task_categories.push_back(tc.as<std::string>());
			}
		}

		/* configs: modern HF format with explicit data_files per split */
		if (root["configs"] && root["configs"].IsSequence()) {
			const auto& configs = root["configs"];
			int chosenIdx = -1;
			for (std::size_t i = 0; i < configs.size(); ++i) {
				const YAML::Node cfg = configs[i];
				if (!config_name.empty()) {
					if (cfg["config_name"] && cfg["config_name"].as<std::string>() == config_name) {
						chosenIdx = static_cast<int>(i);
						break;
					}
				} else {
					chosenIdx = 0; // first config
					break;
				}
			}
			if (chosenIdx < 0) {
				return std::unexpected(
						"parse_dataset_card: config '" + std::string(config_name) + "' not found"
				);
			}
			const YAML::Node cfg = configs[chosenIdx];
			if (cfg["config_name"]) {
				result.config_name = cfg["config_name"].as<std::string>();
			}
			if (cfg["features"] && cfg["features"].IsSequence()) {
				for (const auto& f : cfg["features"]) {
					result.features.push_back(parse_feature(f));
				}
			}
			/* data_files: [{split: train, path: ...}, ...] */
			if (cfg["data_files"] && cfg["data_files"].IsSequence()) {
				for (const auto& df : cfg["data_files"]) {
					DatasetSplit sp;
					if (df["split"]) {
						sp.name = df["split"].as<std::string>();
					}
					if (df["path"]) {
						sp.path = df["path"].as<std::string>();
					}
					result.splits.push_back(std::move(sp));
				}
			}
			return result;
		}

		/* dataset_info: legacy HF format — single mapping or list */
		if (root["dataset_info"]) {
			const auto& di = root["dataset_info"];
			if (di.IsMap()) {
				auto info = parse_dataset_info_node(di);
				result.features = std::move(info.features);
				result.splits   = std::move(info.splits);
				if (result.config_name.empty()) {
					result.config_name = info.config_name;
				}
			} else if (di.IsSequence() && di.size() > 0) {
				/* Multi-config: select by name or fall back to first */
				int chosenIdx = -1;
				for (std::size_t i = 0; i < di.size(); ++i) {
					const YAML::Node entry = di[i];
					if (!config_name.empty()) {
						if (entry["config_name"] && entry["config_name"].as<std::string>() == config_name) {
							chosenIdx = static_cast<int>(i);
							break;
						}
					} else {
						chosenIdx = 0;
						break;
					}
				}
				if (chosenIdx >= 0) {
					const YAML::Node entry = di[chosenIdx];
					auto info = parse_dataset_info_node(entry);
					result.features    = std::move(info.features);
					result.splits      = std::move(info.splits);
					result.config_name = info.config_name;
				}
			}
		}

		return result;
	}

	std::vector<std::filesystem::path> find_split_files(
			const DatasetInfo& /*info*/, std::string_view split, const std::filesystem::path& repo_root
	) {
		const std::filesystem::path dataDir = repo_root / "data";
		if (!std::filesystem::is_directory(dataDir)) {
			return {};
		}

		const std::string prefix = std::string(split) + "-";
		std::vector<std::filesystem::path> result;

		for (const auto& entry : std::filesystem::directory_iterator(dataDir)) {
			if (!entry.is_regular_file()) {
				continue;
			}
			const auto& p   = entry.path();
			const auto  ext = p.extension().string();
			if (ext != ".parquet" && ext != ".arrow") {
				continue;
			}
			const std::string fname = p.filename().string();
			if (fname.starts_with(prefix)) {
				result.push_back(p);
			}
		}

		std::sort(result.begin(), result.end());
		return result;
	}

} // namespace ttm::datasets

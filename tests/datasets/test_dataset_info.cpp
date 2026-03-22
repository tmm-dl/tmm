/**
 * @file test_dataset_info.cpp
 * @brief Unit tests for HuggingFace Dataset Card parsing (parse_dataset_card).
 *
 * @details
 * These tests write temporary README.md fixtures to the filesystem, then
 * verify that parse_dataset_card() correctly extracts features, splits, and
 * task_categories from the YAML frontmatter.
 */

#include <tmm/datasets/dataset_info.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <thread>

#include <catch2/catch_test_macros.hpp>

// =============================================================================
// Test fixture helpers
// =============================================================================

namespace {

	/// RAII helper — creates a temporary directory and removes it on destruction.
	struct TempRepo {
		std::filesystem::path root;

		explicit TempRepo() {
			root = std::filesystem::temp_directory_path() /
				   ("tmm_dataset_test_" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
			std::filesystem::create_directories(root);
		}

		TempRepo(const TempRepo&) = delete;
		TempRepo(TempRepo&&) = delete;
		TempRepo& operator=(const TempRepo&) = delete;
		TempRepo& operator=(TempRepo&&) = delete;

		~TempRepo() {
			std::error_code ec;
			std::filesystem::remove_all(root, ec);
		}

		/// Write content to README.md in the repo root.
		void write_readme(std::string_view content) const {
			std::ofstream ofs(root / "README.md");
			ofs << content;
		}
	};

} // anonymous namespace

// =============================================================================
// Error cases
// =============================================================================

TEST_CASE("parse_dataset_card returns error for missing README", "[dataset_info]") {
	const TempRepo tmp;
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE_FALSE(result.has_value());
	CHECK(result.error().find("README.md") != std::string::npos);
}

TEST_CASE("parse_dataset_card returns error for missing front-matter delimiter", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme("# My Dataset\nNo YAML front matter here.\n");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE_FALSE(result.has_value());
}

// =============================================================================
// Basic parsing
// =============================================================================

TEST_CASE("parse_dataset_card parses a minimal valid card", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme(R"(---
pretty_name: My Test Dataset
---
# My Test Dataset
)");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE(result.has_value());
	CHECK(result->pretty_name == "My Test Dataset");
}

TEST_CASE("parse_dataset_card parses task_categories", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme(R"(---
task_categories:
- text-classification
- token-classification
---
)");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE(result.has_value());
	REQUIRE(result->task_categories.size() == 2);
	CHECK(result->task_categories[0] == "text-classification");
	CHECK(result->task_categories[1] == "token-classification");
}

// =============================================================================
// Features
// =============================================================================

TEST_CASE("parse_dataset_card parses simple scalar features", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme(R"(---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: idx
    dtype: int32
---
)");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE(result.has_value());
	REQUIRE(result->features.size() == 2);
	CHECK(result->features[0].name == "text");
	CHECK(result->features[0].dtype == "string");
	CHECK(result->features[0].kind == tmm::datasets::FeatureKind::Scalar);
	CHECK(result->features[1].name == "idx");
	CHECK(result->features[1].kind == tmm::datasets::FeatureKind::Scalar);
}

TEST_CASE("parse_dataset_card parses class_label feature", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme(R"(---
dataset_info:
  features:
  - name: label
    dtype:
      class_label:
        names:
        - positive
        - negative
        - neutral
---
)");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE(result.has_value());
	REQUIRE(result->features.size() == 1);
	const auto& feat = result->features[0];
	CHECK(feat.name == "label");
	CHECK(feat.kind == tmm::datasets::FeatureKind::ClassLabel);
	REQUIRE(feat.class_names.size() == 3);
	CHECK(feat.class_names[0] == "positive");
	CHECK(feat.class_names[1] == "negative");
	CHECK(feat.class_names[2] == "neutral");
}

TEST_CASE("parse_dataset_card parses sequence feature", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme(R"(---
dataset_info:
  features:
  - name: tokens
    sequence: string
---
)");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE(result.has_value());
	REQUIRE(result->features.size() == 1);
	const auto& feat = result->features[0];
	CHECK(feat.name == "tokens");
	CHECK(feat.kind == tmm::datasets::FeatureKind::Sequence);
}

// =============================================================================
// Splits
// =============================================================================

TEST_CASE("parse_dataset_card parses splits with num_examples", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme(R"(---
dataset_info:
  splits:
  - name: train
    num_examples: 5000
    num_bytes: 102400
  - name: test
    num_examples: 1000
    num_bytes: 20480
---
)");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE(result.has_value());
	REQUIRE(result->splits.size() == 2);
	CHECK(result->splits[0].name == "train");
	CHECK(result->splits[0].num_examples == 5000);
	CHECK(result->splits[0].num_bytes == 102400);
	CHECK(result->splits[1].name == "test");
	CHECK(result->splits[1].num_examples == 1000);
}

// =============================================================================
// Multi-config dataset_info (list form)
// =============================================================================

TEST_CASE("parse_dataset_card uses first config when dataset_info is a list", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme(R"(---
dataset_info:
- config_name: en
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_examples: 1000
- config_name: fr
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_examples: 800
---
)");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE(result.has_value());
	CHECK(result->config_name == "en");
	REQUIRE(result->splits.size() == 1);
	CHECK(result->splits[0].num_examples == 1000);
}

// =============================================================================
// Full card (combined)
// =============================================================================

TEST_CASE("parse_dataset_card parses a complete HF-style card", "[dataset_info]") {
	const TempRepo tmp;
	tmp.write_readme(R"(---
pretty_name: IMDB Movie Reviews
task_categories:
- text-classification
dataset_info:
  config_name: plain_text
  features:
  - name: text
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
        - neg
        - pos
  splits:
  - name: train
    num_bytes: 33432835
    num_examples: 25000
  - name: test
    num_bytes: 32650697
    num_examples: 25000
---
# IMDB Movie Reviews
)");
	const auto result = tmm::datasets::parse_dataset_card(tmp.root);
	REQUIRE(result.has_value());
	CHECK(result->pretty_name == "IMDB Movie Reviews");
	CHECK(result->config_name == "plain_text");
	REQUIRE(result->task_categories.size() == 1);
	CHECK(result->task_categories[0] == "text-classification");
	REQUIRE(result->features.size() == 2);
	CHECK(result->features[0].name == "text");
	CHECK(result->features[0].kind == tmm::datasets::FeatureKind::Scalar);
	CHECK(result->features[1].name == "label");
	CHECK(result->features[1].kind == tmm::datasets::FeatureKind::ClassLabel);
	REQUIRE(result->features[1].class_names.size() == 2);
	REQUIRE(result->splits.size() == 2);
	CHECK(result->splits[0].name == "train");
	CHECK(result->splits[0].num_examples == 25000);
}

// =============================================================================
// find_split_files
// =============================================================================

TEST_CASE("find_split_files returns empty when data/ directory is missing", "[dataset_info]") {
	const TempRepo tmp;
	const tmm::datasets::DatasetInfo info;
	const auto files = tmm::datasets::find_split_files(info, "train", tmp.root);
	CHECK(files.empty());
}

TEST_CASE("find_split_files finds parquet shards for a split", "[dataset_info]") {
	const TempRepo tmp;
	std::filesystem::create_directories(tmp.root / "data");

	/* Create shard files */
	for (const auto* name :
		 {"train-00000-of-00002.parquet", "train-00001-of-00002.parquet", "test-00000-of-00001.parquet"}) {
		std::ofstream{tmp.root / "data" / name};
	}

	const tmm::datasets::DatasetInfo info;
	const auto trainFiles = tmm::datasets::find_split_files(info, "train", tmp.root);
	REQUIRE(trainFiles.size() == 2);
	/* Verify sorting */
	CHECK(trainFiles[0].filename().string() < trainFiles[1].filename().string());

	const auto testFiles = tmm::datasets::find_split_files(info, "test", tmp.root);
	CHECK(testFiles.size() == 1);

	const auto valFiles = tmm::datasets::find_split_files(info, "validation", tmp.root);
	CHECK(valFiles.empty());
}

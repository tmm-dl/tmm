/**
 * @file loader.hpp
 * @brief Config file loading, deep-merge, override application, and validation.
 *
 * @details
 * Typical usage:
 * @code{.cpp}
 * // Single file
 * auto cfg = ttm::conf::load_config("train.yml").value();
 *
 * // Multiple files (deep-merged left-to-right) + CLI --set overrides
 * std::vector<std::filesystem::path> files = {"base.yml", "experiment.yml"};
 * std::vector<std::string> overrides = {"optimizer.lr=5e-5", "training.epochs=20"};
 * auto cfg = ttm::conf::load_config(files, overrides).value();
 * @endcode
 *
 * ### Merge semantics
 * Files are deep-merged left-to-right:
 * - Scalar values and sequences: right side wins.
 * - Mapping (dict) keys: recursively merged; right wins on leaf conflicts.
 *
 * ### Override syntax
 * `--set` values use dot-separated key paths: `optimizer.lr=1e-4`.
 * Intermediate maps are created if absent.
 *
 * ### Environment interpolation
 * String scalars may contain `${VAR_NAME}` references which are replaced
 * with the corresponding environment variable value (empty string if unset).
 */

#pragma once

#include <ttm/conf/config.hpp>
#include <ttm/compat/expected.hpp>

#include <filesystem>
#include <span>
#include <string>

namespace ttm::conf {

	/**
	 * @brief Load and merge one or more YAML config files, then apply overrides.
	 *
	 * @param files         YAML config files to load and deep-merge.
	 * @param set_overrides `key=value` pairs from CLI `--set` flags.
	 * @return Resolved @ref TrainingConfig on success, or an error string.
	 */
	[[nodiscard]] std::expected<TrainingConfig, std::string>
	load_config(
		std::span<const std::filesystem::path> files,
		std::span<const std::string>           set_overrides = {}
	);

	/// Convenience overload for a single config file.
	[[nodiscard]] std::expected<TrainingConfig, std::string>
	load_config(
		const std::filesystem::path& file,
		std::span<const std::string> set_overrides = {}
	);

} // namespace ttm::conf

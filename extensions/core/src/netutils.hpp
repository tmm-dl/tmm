/**
 * @file netutils.hpp
 * @brief Network utility helpers for the core plugin.
 *
 * @details
 * Provides git-lfs pointer detection and cpr-based file downloading.
 * Used by git_source.cpp to transparently resolve lfs-tracked dataset files.
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

/**
 * @brief Return true if @p fp starts with the git-lfs pointer magic string.
 *
 * Rewinds the file to its original position before returning so the caller
 * can continue reading the file normally on a false result.
 */
bool isLfsPointer(std::FILE* fp);

/**
 * @brief Download @p url to @p dest via cpr.
 *
 * Creates parent directories as needed.  On failure the partial download is
 * removed and a descriptive message is written to @p err.
 *
 * @return true on success, false with @p err filled on failure.
 */
bool httpDownload(const std::string& url, const std::filesystem::path& dest, char* err, uint32_t err_cap);

/**
 * @brief Resolve a git-lfs pointer to real file content.
 *
 * Downloads from `<git_url>/resolve/<ref>/<subpath>` and caches the result
 * at `<localRepo>/.tmm_lfs/<subpath>`.  Subsequent calls return the cached
 * file directly without re-downloading.
 *
 * @return Opened `FILE*` on success, `nullptr` with @p err filled on failure.
 */
std::FILE*
lfsOpen(const std::string& git_url, const std::string& ref, const std::string& subpath,
		const std::filesystem::path& localRepo, char* err, uint32_t err_cap);

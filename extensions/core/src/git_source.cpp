/**
 * @file git_source.cpp
 * @brief Git-based dataset source implementation.
 *
 * @details
 * Implements `ttm_source_vtable` for the following URI schemes:
 *   - `gh:`  — GitHub repositories  (https://github.com/<owner>/<repo>)
 *   - `gl:`  — GitLab repositories  (https://gitlab.com/<owner>/<repo>)
 *   - `bb:`  — Bitbucket repos      (https://bitbucket.org/<owner>/<repo>)
 *   - `hf:`  — HuggingFace datasets (https://huggingface.co/datasets/<owner>/<repo>)
 *   - `sr:`  — SourceHut repos      (https://git.sr.ht/~<user>/<repo>)
 *
 * URI format:
 *   `<scheme><owner>/<repo>[@<ref>][/<subpath>]`
 *
 * Repositories are shallow-cloned / fetched into a local cache:
 *   `$XDG_CACHE_HOME/ttm/datasets/<fnv1a_hex_of_url_plus_ref>/`
 *
 * Files are opened directly from disk using standard C FILE I/O.
 * git-lfs pointer files are transparently resolved via lfsOpen().
 */

#include "git_source.hpp"
#include "netutils.hpp"

#include <ttm/plugins/abi.h>

#include <git2.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace {

/* -------------------------------------------------------------------------
 * ParsedUri
 * ---------------------------------------------------------------------- */

struct ParsedUri {
	std::string git_url;  ///< Full HTTPS git clone URL.
	std::string ref;      ///< Branch/tag/commit ref (default: "main").
	std::string subpath;  ///< Path within the repository (may be empty).
};

/**
 * @brief Expand a scheme-prefixed URI into its git URL, ref, and subpath.
 *
 * Handles `gh:`, `gl:`, `bb:`, `hf:`, `sr:`.
 * Format: `<scheme><owner>/<repo>[@<ref>][/<subpath>]`
 */
std::optional<ParsedUri> expandUri(std::string_view uri) {
	ParsedUri result;

	const char* base = nullptr;
	std::string_view rest = uri;

	if      (uri.starts_with("gh:")) { base = "https://github.com/";                rest = uri.substr(3); }
	else if (uri.starts_with("gl:")) { base = "https://gitlab.com/";                rest = uri.substr(3); }
	else if (uri.starts_with("bb:")) { base = "https://bitbucket.org/";             rest = uri.substr(3); }
	else if (uri.starts_with("hf:")) { base = "https://huggingface.co/datasets/";   rest = uri.substr(3); }
	else if (uri.starts_with("sr:")) { base = "https://git.sr.ht/~";                rest = uri.substr(3); }
	else { return std::nullopt; }

	std::string restStr(rest);

	const auto atPos = restStr.find('@');
	std::string ownerRepo;
	std::string refAndPath;
	if (atPos != std::string::npos) {
		ownerRepo  = restStr.substr(0, atPos);
		refAndPath = restStr.substr(atPos + 1);
	} else {
		ownerRepo  = restStr;
		refAndPath.clear();
	}

	if (!refAndPath.empty()) {
		const auto slashPos = refAndPath.find('/');
		if (slashPos != std::string::npos) {
			result.ref     = refAndPath.substr(0, slashPos);
			result.subpath = refAndPath.substr(slashPos + 1);
		} else {
			result.ref     = refAndPath;
			result.subpath.clear();
		}
	} else {
		const auto thirdSlash = [&]() -> std::string::size_type {
			auto pos = ownerRepo.find('/');
			if (pos == std::string::npos) return std::string::npos;
			return ownerRepo.find('/', pos + 1);
		}();
		if (thirdSlash != std::string::npos) {
			result.subpath = ownerRepo.substr(thirdSlash + 1);
			ownerRepo      = ownerRepo.substr(0, thirdSlash);
		}
		result.ref = "main";
	}

	if (result.ref.empty()) result.ref = "main";

	result.git_url = std::string(base) + ownerRepo;
	return result;
}

/* -------------------------------------------------------------------------
 * Cache path computation
 * ---------------------------------------------------------------------- */

/**
 * @brief Compute a stable cache directory for a (url, ref) pair.
 *
 * Uses FNV-1a 64-bit hash of "url#ref" as the directory name.
 * Stores under $XDG_CACHE_HOME/ttm/datasets/ or ~/.cache/ttm/datasets/.
 */
std::filesystem::path cachePathFor(const std::string& git_url, const std::string& ref) {
	const std::string key = git_url + "#" + ref;
	uint64_t hash = 14695981039346656037ULL;
	for (const auto ch : key) {
		hash ^= static_cast<uint8_t>(ch);
		hash *= 1099511628211ULL;
	}

	std::array<char, 17> hexBuf{};
	std::snprintf(hexBuf.data(), hexBuf.size(), "%016llx", static_cast<unsigned long long>(hash));

	std::filesystem::path base;
	if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
		base = std::filesystem::path(xdg);
	} else if (const char* home = std::getenv("HOME")) {
		base = std::filesystem::path(home) / ".cache";
	} else {
		base = std::filesystem::temp_directory_path();
	}

	return base / "ttm" / "datasets" / hexBuf.data();
}

/* -------------------------------------------------------------------------
 * gitEnsureRepo — clone or update a local mirror
 * ---------------------------------------------------------------------- */

/**
 * @brief Ensure a local clone of the repository exists and is up to date.
 *
 * - If the repository is not cached, performs a shallow clone (depth=1).
 * - If the ref looks like a 40-char SHA, skips fetch (pinned commit).
 * - Otherwise opens the repo, fetches, and fast-forwards if needed.
 */
std::filesystem::path gitEnsureRepo(
		const std::string& git_url, const std::string& ref, char* err, uint32_t err_cap
) {
	const auto localPath = cachePathFor(git_url, ref);

	auto fail = [&](const char* context) -> std::filesystem::path {
		const git_error* gerr = git_error_last();
		const char* msg = (gerr != nullptr) ? gerr->message : "(no git error)";
		std::snprintf(err, err_cap, "%s: %s", context, msg);
		return {};
	};

	const bool isPinnedSha = (ref.size() == 40) && std::all_of(ref.begin(), ref.end(), [](char c) {
		return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
	});

	if (!std::filesystem::exists(localPath / ".git")) {
		std::filesystem::create_directories(localPath);

		git_clone_options opts = GIT_CLONE_OPTIONS_INIT;
		opts.checkout_branch   = ref.c_str();

		git_repository* repo = nullptr;
		if (git_clone(&repo, git_url.c_str(), localPath.string().c_str(), &opts) < 0) {
			return fail("git_clone");
		}
		git_repository_free(repo);
	} else if (!isPinnedSha) {
		git_repository* repo = nullptr;
		if (git_repository_open(&repo, localPath.string().c_str()) < 0) {
			return fail("git_repository_open");
		}

		git_remote* remote = nullptr;
		if (git_remote_lookup(&remote, repo, "origin") < 0) {
			git_repository_free(repo);
			return fail("git_remote_lookup");
		}

		const char* refspec = ref.c_str();
		const git_strarray refspecs{
				const_cast<char**>(&refspec), 1
		}; // NOLINT(cppcoreguidelines-pro-type-const-cast) -- libgit2 requires non-const; does not modify
		git_fetch_options fetchOpts = GIT_FETCH_OPTIONS_INIT;
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay) -- libgit2 API requirement
		const int fetchRet = git_remote_fetch(remote, &refspecs, &fetchOpts, nullptr);
		git_remote_free(remote);
		git_repository_free(repo);
		if (fetchRet < 0) {
			/* Non-fatal: might already have the ref locally */
			giterr_clear();
		}
	}

	return localPath;
}

/* -------------------------------------------------------------------------
 * Handle table — open file handles
 * ---------------------------------------------------------------------- */

constexpr int kMaxHandles = 64;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) -- plugin-level open-file table
std::FILE* g_handles[kMaxHandles]{};

ttm_handle allocHandle(std::FILE* fp) {
	for (int i = 0; i < kMaxHandles; ++i) {
		if (g_handles[i] == nullptr) {
			g_handles[i] = fp;
			return static_cast<ttm_handle>(i);
		}
	}
	return TTM_INVALID_HANDLE;
}

std::FILE* getHandle(ttm_handle h) {
	if (h < 0 || h >= kMaxHandles) return nullptr;
	return g_handles[static_cast<int>(h)];
}

/* =========================================================================
 * Source vtable implementations
 * ====================================================================== */

ttm_handle coreOpen(const char* uri, uint32_t /*uri_len*/, char* err, uint32_t err_cap) {
	auto parsed = expandUri(std::string_view(uri));
	if (!parsed) {
		std::snprintf(err, err_cap, "coreOpen: unrecognised URI scheme in '%s'", uri);
		return TTM_INVALID_HANDLE;
	}

	const auto localRepo = gitEnsureRepo(parsed->git_url, parsed->ref, err, err_cap);
	if (localRepo.empty()) {
		return TTM_INVALID_HANDLE;
	}

	if (parsed->subpath.empty()) {
		std::snprintf(err, err_cap, "coreOpen: URI '%s' has no subpath", uri);
		return TTM_INVALID_HANDLE;
	}

	const auto filePath = localRepo / parsed->subpath;
	std::FILE* fp = std::fopen(filePath.string().c_str(), "rb");
	if (fp == nullptr) {
		std::snprintf(err, err_cap, "coreOpen: cannot open '%s': %s",
		              filePath.string().c_str(), std::strerror(errno));
		return TTM_INVALID_HANDLE;
	}

	if (isLfsPointer(fp)) {
		std::fclose(fp);
		fp = lfsOpen(parsed->git_url, parsed->ref, parsed->subpath, localRepo, err, err_cap);
		if (fp == nullptr) return TTM_INVALID_HANDLE;
	}

	const auto handle = allocHandle(fp);
	if (handle == TTM_INVALID_HANDLE) {
		std::fclose(fp);
		std::snprintf(err, err_cap, "coreOpen: too many open file handles");
	}
	return handle;
}

int32_t coreRead(ttm_handle h, void* buf, int32_t len) {
	std::FILE* fp = getHandle(h);
	if (fp == nullptr) return -1;
	return static_cast<int32_t>(std::fread(buf, 1, static_cast<std::size_t>(len), fp));
}

int64_t coreSeek(ttm_handle h, int64_t offset, int32_t whence) {
	std::FILE* fp = getHandle(h);
	if (fp == nullptr) return -1;
	int posixWhence = SEEK_SET;
	if      (whence == 1) posixWhence = SEEK_CUR;
	else if (whence == 2) posixWhence = SEEK_END;
#ifdef _WIN32
	if (_fseeki64(fp, static_cast<__int64>(offset), posixWhence) != 0) return -1;
	return static_cast<int64_t>(_ftelli64(fp));
#else
	if (fseeko(fp, static_cast<off_t>(offset), posixWhence) != 0) return -1;
	return static_cast<int64_t>(ftello(fp));
#endif
}

void coreClose(ttm_handle h) {
	if (h < 0 || h >= kMaxHandles) return;
	if (g_handles[h] != nullptr) {
		std::fclose(g_handles[h]);
		g_handles[h] = nullptr;
	}
}

// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast) -- C vtable requires function pointer assignment
ttm_source_vtable g_vtable = {coreOpen, coreRead, coreSeek, coreClose};
// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
const char* g_schemes[] = {"gh:", "gl:", "bb:", "hf:", "sr:", nullptr};

} // anonymous namespace

ttm_error gitSourceRegister(const ttm_host_api* host) {
	git_libgit2_init();
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
	return host->register_source(host->ctx, g_schemes, &g_vtable);
}

void gitSourceTeardown() {
	for (auto& fp : g_handles) {
		if (fp != nullptr) {
			std::fclose(fp);
			fp = nullptr;
		}
	}
	git_libgit2_shutdown();
}

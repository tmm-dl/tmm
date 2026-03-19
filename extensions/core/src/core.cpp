/**
 * @file core.cpp
 * @brief TTM core native plugin — git-based dataset source provider.
 *
 * @details
 * Implements the `ttm_source_vtable` for the following URI schemes:
 *   - `gh:`  — GitHub repositories  (https://github.com/<owner>/<repo>)
 *   - `gl:`  — GitLab repositories  (https://gitlab.com/<owner>/<repo>)
 *   - `bb:`  — Bitbucket repos      (https://bitbucket.org/<owner>/<repo>)
 *   - `hf:`  — HuggingFace datasets (https://huggingface.co/datasets/<owner>/<repo>)
 *   - `sr:`  — SourceHut repos      (https://git.sr.ht/~<user>/<repo>)
 *
 * URI format:
 *   `<scheme><owner>/<repo>[@<ref>][/<subpath>]`
 *   e.g. `hf:ylecun/mnist@main/data/train-00000-of-00001.parquet`
 *
 * Repositories are shallow-cloned / fetched into a local cache:
 *   `$XDG_CACHE_HOME/ttm/datasets/<sha256_hex_of_url_plus_ref>/`
 *   (falls back to `~/.cache/ttm/datasets/…`)
 *
 * Once the repo is in the local cache, files are opened directly from disk
 * using standard C FILE I/O.
 */

#include <ttm/plugins/abi.h>
#include <ttm_core_export.h>

#include <curl/curl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <git2.h>

/* ============================================================================
 * Compile-time checks
 * ========================================================================= */

static_assert(sizeof(ttm_handle) == sizeof(int64_t), "ttm_handle must be 64 bits");

/* ============================================================================
 * Internal types and helpers
 * ========================================================================= */

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
	std::optional<ParsedUri> expand_uri(std::string_view uri) {
		ParsedUri result;

		/* Determine scheme and base URL */
		const char* base = nullptr;
		std::string_view rest = uri;
		const bool isSr = uri.starts_with("sr:");

		if (uri.starts_with("gh:")) {
			base = "https://github.com/";
			rest = uri.substr(3);
		} else if (uri.starts_with("gl:")) {
			base = "https://gitlab.com/";
			rest = uri.substr(3);
		} else if (uri.starts_with("bb:")) {
			base = "https://bitbucket.org/";
			rest = uri.substr(3);
		} else if (uri.starts_with("hf:")) {
			base = "https://huggingface.co/datasets/";
			rest = uri.substr(3);
		} else if (isSr) {
			base = "https://git.sr.ht/~";
			rest = uri.substr(3);
		} else {
			return std::nullopt;
		}

		/* Split owner/repo[@ref][/subpath] */
		/* First, isolate owner/repo (before first '@' or third '/' after scheme) */
		std::string restStr(rest);

		/* Extract ref if present */
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

		/* Split ref from subpath: first '/' in refAndPath separates ref from subpath */
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
			/* No ref in URI — extract subpath from ownerRepo if more than 2 components */
			const auto thirdSlash = [&]() -> std::string::size_type {
				auto pos = ownerRepo.find('/');
				if (pos == std::string::npos) {
					return std::string::npos;
				}
				return ownerRepo.find('/', pos + 1);
			}();
			if (thirdSlash != std::string::npos) {
				result.subpath = ownerRepo.substr(thirdSlash + 1);
				ownerRepo      = ownerRepo.substr(0, thirdSlash);
			}
			result.ref = "main";
		}

		if (result.ref.empty()) {
			result.ref = "main";
		}

		result.git_url = std::string(base) + ownerRepo;
		return result;
	}

	/* -------------------------------------------------------------------------
	 * Cache path computation
	 * ---------------------------------------------------------------------- */

	/**
	 * @brief Compute a stable cache directory for a (url, ref) pair.
	 *
	 * Uses a simple FNV-1a 64-bit hash of "url#ref" as the directory name.
	 * Stores under $XDG_CACHE_HOME/ttm/datasets/ or ~/.cache/ttm/datasets/.
	 */
	std::filesystem::path cache_path_for(const std::string& git_url, const std::string& ref) {
		/* FNV-1a 64-bit hash */
		const std::string key = git_url + "#" + ref;
		uint64_t hash = 14695981039346656037ULL;
		for (const auto ch : key) {
			hash ^= static_cast<uint8_t>(ch);
			hash *= 1099511628211ULL;
		}

		/* Format as 16-char hex */
		std::array<char, 17> hexBuf{};
		std::snprintf(hexBuf.data(), hexBuf.size(), "%016llx", static_cast<unsigned long long>(hash));

		/* Base directory */
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
	 * git_ensure_repo — clone or update a local mirror
	 * ---------------------------------------------------------------------- */

	/**
	 * @brief Ensure a local clone of the repository exists and is up to date.
	 *
	 * - If the repository is not cached, performs a shallow clone (depth=1).
	 * - If the ref looks like a 40-char SHA, skips fetch (pinned commit).
	 * - Otherwise opens the repo, fetches, and fast-forwards if needed.
	 *
	 * @param[in]  git_url  Full HTTPS clone URL.
	 * @param[in]  ref      Branch / tag / SHA ref.
	 * @param[out] err      Error buffer for diagnostic messages.
	 * @param[in]  err_cap  Error buffer capacity.
	 * @return Local repository root path on success, or empty path on failure.
	 */
	std::filesystem::path git_ensure_repo(
			const std::string& git_url, const std::string& ref, char* err, uint32_t err_cap
	) {
		const auto localPath = cache_path_for(git_url, ref);

		auto fail = [&](const char* context) -> std::filesystem::path {
			const git_error* gerr = git_error_last();
			const char* msg = (gerr != nullptr) ? gerr->message : "(no git error)";
			std::snprintf(err, err_cap, "%s: %s", context, msg);
			return {};
		};

		/* Is a 40-char hex SHA? (pinned commit — skip fetch) */
		const bool isPinnedSha = (ref.size() == 40) && std::all_of(ref.begin(), ref.end(), [](char c) {
			return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
		});

		if (!std::filesystem::exists(localPath / ".git")) {
			/* Clone */
			std::filesystem::create_directories(localPath);

			git_clone_options opts = GIT_CLONE_OPTIONS_INIT;
			opts.checkout_branch   = ref.c_str();

			/* Shallow clone via fetch depth — only available in newer libgit2 */
			/* opts.fetch_opts.depth = 1; // libgit2 1.x supports this */

			git_repository* repo = nullptr;
			if (git_clone(&repo, git_url.c_str(), localPath.string().c_str(), &opts) < 0) {
				return fail("git_clone");
			}
			git_repository_free(repo);
		} else if (!isPinnedSha) {
			/* Open and fetch */
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

	struct FileEntry {
		std::FILE* fp = nullptr;
	};

	constexpr int kMaxHandles = 64;
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) -- plugin-level open-file table; no safer alternative without heap allocation
	FileEntry g_handles[kMaxHandles]{};

	ttm_handle alloc_handle(std::FILE* fp) {
		for (int i = 0; i < kMaxHandles; ++i) {
			if (g_handles[i].fp == nullptr) {
				g_handles[i].fp = fp;
				return static_cast<ttm_handle>(i);
			}
		}
		return TTM_INVALID_HANDLE;
	}

	FileEntry* get_handle(ttm_handle h) {
		if (h < 0 || h >= kMaxHandles) {
			return nullptr;
		}
		return &g_handles[static_cast<int>(h)];
	}

	/* =========================================================================
	 * git-lfs helpers
	 * ====================================================================== */

	/**
	 * @brief Return true if `fp` starts with the git-lfs pointer magic.
	 *
	 * A git-lfs pointer file begins with the line:
	 *   `version https://git-lfs.github.com/spec/1`
	 * Rewinds the file position before returning.
	 */
	bool is_lfs_pointer(std::FILE* fp) {
		static constexpr std::string_view kLfsMagic = "version https://git-lfs.github.com/spec/v1";
		std::array<char, 42> buf{};
		const std::size_t n = std::fread(buf.data(), 1, kLfsMagic.size(), fp);
		std::rewind(fp);
		return n == kLfsMagic.size() && std::memcmp(buf.data(), kLfsMagic.data(), kLfsMagic.size()) == 0;
	}

	/** @brief libcurl write callback: writes received bytes to a FILE*. */
	std::size_t curl_write_cb(const char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
		return std::fwrite(ptr, size, nmemb, static_cast<std::FILE*>(userdata));
	}

	/**
	 * @brief Download `url` to `dest` via libcurl.
	 * @return true on success; false with `err` filled on failure.
	 */
	bool curl_download(const std::string& url, const std::filesystem::path& dest, char* err, uint32_t err_cap) {
		std::filesystem::create_directories(dest.parent_path());

		std::FILE* out = std::fopen(dest.string().c_str(), "wb");
		if (out == nullptr) {
			std::snprintf(err, err_cap, "curl_download: cannot create '%s': %s",
			              dest.string().c_str(), std::strerror(errno));
			return false;
		}

		CURL* curl = curl_easy_init();
		if (curl == nullptr) {
			std::fclose(out);
			std::snprintf(err, err_cap, "curl_download: curl_easy_init failed");
			return false;
		}

		curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
		curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
		curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

		const CURLcode rc = curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		std::fclose(out);

		if (rc != CURLE_OK) {
			std::filesystem::remove(dest);
			std::snprintf(err, err_cap, "curl_download: %s: %s", url.c_str(), curl_easy_strerror(rc));
			return false;
		}
		return true;
	}

	/**
	 * @brief Resolve a git-lfs pointer to the real file content.
	 *
	 * Downloads the real content from `<git_url>/resolve/<ref>/<subpath>`
	 * and caches it at `localRepo/.ttm_lfs/<subpath>` — inside the repo's
	 * existing XDG cache directory — so all data for one repo stays together.
	 * On subsequent opens the cached file is returned directly.
	 *
	 * @return Opened FILE* on success, nullptr with `err` filled on failure.
	 */
	std::FILE* lfs_open(
			const std::string& git_url, const std::string& ref, const std::string& subpath,
			const std::filesystem::path& localRepo, char* err, uint32_t err_cap
	) {
		/* Stable cache path inside the repo's XDG directory */
		const auto cachePath = localRepo / ".ttm_lfs" / subpath;

		if (!std::filesystem::exists(cachePath)) {
			/* HuggingFace resolve URL: git_url/resolve/ref/subpath */
			const std::string resolveUrl = git_url + "/resolve/" + ref + "/" + subpath;
			if (!curl_download(resolveUrl, cachePath, err, err_cap)) {
				return nullptr;
			}
		}

		std::FILE* fp = std::fopen(cachePath.string().c_str(), "rb");
		if (fp == nullptr) {
			std::snprintf(err, err_cap, "lfs_open: cannot open '%s': %s",
			              cachePath.string().c_str(), std::strerror(errno));
		}
		return fp;
	}

	/* =========================================================================
	 * Source vtable implementations
	 * ====================================================================== */

	ttm_handle core_open(const char* uri, uint32_t /*uri_len*/, char* err, uint32_t err_cap) {
		auto parsed = expand_uri(std::string_view(uri));
		if (!parsed) {
			std::snprintf(err, err_cap, "core_open: unrecognised URI scheme in '%s'", uri);
			return TTM_INVALID_HANDLE;
		}

		const auto localRepo = git_ensure_repo(parsed->git_url, parsed->ref, err, err_cap);
		if (localRepo.empty()) {
			return TTM_INVALID_HANDLE; /* err already filled by git_ensure_repo */
		}

		if (parsed->subpath.empty()) {
			std::snprintf(err, err_cap, "core_open: URI '%s' has no subpath", uri);
			return TTM_INVALID_HANDLE;
		}

		const auto filePath = localRepo / parsed->subpath;
		std::FILE* fp = std::fopen(filePath.string().c_str(), "rb");
		if (fp == nullptr) {
			std::snprintf(
					err, err_cap, "core_open: cannot open '%s': %s", filePath.string().c_str(), std::strerror(errno)
			);
			return TTM_INVALID_HANDLE;
		}

		/* Detect git-lfs pointer and download the real content if needed */
		if (is_lfs_pointer(fp)) {
			std::fclose(fp);
			fp = lfs_open(parsed->git_url, parsed->ref, parsed->subpath, localRepo, err, err_cap);
			if (fp == nullptr) {
				return TTM_INVALID_HANDLE;
			}
		}

		const auto handle = alloc_handle(fp);
		if (handle == TTM_INVALID_HANDLE) {
			std::fclose(fp);
			std::snprintf(err, err_cap, "core_open: too many open file handles");
		}
		return handle;
	}

	int32_t core_read(ttm_handle h, void* buf, int32_t len) {
		auto* entry = get_handle(h);
		if (entry == nullptr || entry->fp == nullptr) {
			return -1;
		}
		return static_cast<int32_t>(std::fread(buf, 1, static_cast<std::size_t>(len), entry->fp));
	}

	int64_t core_seek(ttm_handle h, int64_t offset, int32_t whence) {
		auto* entry = get_handle(h);
		if (entry == nullptr || entry->fp == nullptr) {
			return -1;
		}
		int posixWhence = SEEK_SET;
		if (whence == 1) {
			posixWhence = SEEK_CUR;
		} else if (whence == 2) {
			posixWhence = SEEK_END;
		}
#ifdef _WIN32
		if (_fseeki64(entry->fp, static_cast<__int64>(offset), posixWhence) != 0) {
			return -1;
		}
		return static_cast<int64_t>(_ftelli64(entry->fp));
#else
		if (fseeko(entry->fp, static_cast<off_t>(offset), posixWhence) != 0) {
			return -1;
		}
		return static_cast<int64_t>(ftello(entry->fp));
#endif
	}

	void core_close(ttm_handle h) {
		auto* entry = get_handle(h);
		if (entry != nullptr && entry->fp != nullptr) {
			std::fclose(entry->fp);
			entry->fp = nullptr;
		}
	}

	/* =========================================================================
	 * Plugin metadata and vtables
	 * ====================================================================== */

	ttm_plugin_info g_info = {
			TTM_ABI_VERSION,
			"core",
			"0.1.0",
			"Dataset sources: gh: gl: bb: hf: sr:"
	};

	// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast) -- C vtable requires function pointer assignment; types are compatible
	ttm_source_vtable g_vtable = {core_open, core_read, core_seek, core_close};
	// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) -- C ABI null-terminated scheme array
	const char* g_schemes[] = {"gh:", "gl:", "bb:", "hf:", "sr:", nullptr};

} // anonymous namespace

/* ============================================================================
 * Required plugin exports
 * ========================================================================= */

extern "C" {

TTM_CORE_EXPORT ttm_plugin_info* ttm_plugin_get_info(void) {
	return &g_info;
}

TTM_CORE_EXPORT ttm_error ttm_plugin_init(const ttm_host_api* host, const char* /*cfg*/, uint32_t /*len*/) {
	git_libgit2_init();
	// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) -- C ABI null-terminated array
	return host->register_source(host->ctx, g_schemes, &g_vtable);
}

TTM_CORE_EXPORT void ttm_plugin_teardown(void) {
	/* Close any leftover file handles */
	for (auto& entry : g_handles) {
		if (entry.fp != nullptr) {
			std::fclose(entry.fp);
			entry.fp = nullptr;
		}
	}
	git_libgit2_shutdown();
}

} // extern "C"

/**
 * @file netutils.cpp
 * @brief Network utility helpers: git-lfs detection and libcurl downloads.
 */

#include "netutils.hpp"

#include <curl/curl.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string_view>

namespace {

/** @brief libcurl write callback — writes received bytes straight to a FILE*. */
std::size_t curlWriteCb(const char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
	return std::fwrite(ptr, size, nmemb, static_cast<std::FILE*>(userdata));
}

} // anonymous namespace

bool isLfsPointer(std::FILE* fp) {
	static constexpr std::string_view kLfsMagic = "version https://git-lfs.github.com/spec/v1";
	std::array<char, 42> buf{};
	const std::size_t n = std::fread(buf.data(), 1, kLfsMagic.size(), fp);
	std::rewind(fp);
	return n == kLfsMagic.size() && std::memcmp(buf.data(), kLfsMagic.data(), kLfsMagic.size()) == 0;
}

bool curlDownload(const std::string& url, const std::filesystem::path& dest,
                  char* err, uint32_t err_cap) {
	std::filesystem::create_directories(dest.parent_path());

	std::FILE* out = std::fopen(dest.string().c_str(), "wb");
	if (out == nullptr) {
		std::snprintf(err, err_cap, "curlDownload: cannot create '%s': %s",
		              dest.string().c_str(), std::strerror(errno));
		return false;
	}

	CURL* curl = curl_easy_init();
	if (curl == nullptr) {
		std::fclose(out);
		std::snprintf(err, err_cap, "curlDownload: curl_easy_init failed");
		return false;
	}

	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

	const CURLcode rc = curl_easy_perform(curl);
	curl_easy_cleanup(curl);
	std::fclose(out);

	if (rc != CURLE_OK) {
		std::filesystem::remove(dest);
		std::snprintf(err, err_cap, "curlDownload: %s: %s", url.c_str(), curl_easy_strerror(rc));
		return false;
	}
	return true;
}

std::FILE* lfsOpen(const std::string& git_url, const std::string& ref,
                   const std::string& subpath,
                   const std::filesystem::path& localRepo,
                   char* err, uint32_t err_cap) {
	const auto cachePath = localRepo / ".ttm_lfs" / subpath;

	if (!std::filesystem::exists(cachePath)) {
		const std::string resolveUrl = git_url + "/resolve/" + ref + "/" + subpath;
		if (!curlDownload(resolveUrl, cachePath, err, err_cap)) {
			return nullptr;
		}
	}

	std::FILE* fp = std::fopen(cachePath.string().c_str(), "rb");
	if (fp == nullptr) {
		std::snprintf(err, err_cap, "lfsOpen: cannot open '%s': %s",
		              cachePath.string().c_str(), std::strerror(errno));
	}
	return fp;
}

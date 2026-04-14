/**
 * @file netutils.cpp
 * @brief Network utility helpers: git-lfs detection and HTTP downloads via cpr.
 */

#include "netutils.hpp"

#include <cpr/cpr.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string_view>

bool isLfsPointer(std::FILE* fp) {
	static constexpr std::string_view kLfsMagic = "version https://git-lfs.github.com/spec/v1";
	std::array<char, 42> buf{};
	const std::size_t n = std::fread(buf.data(), 1, kLfsMagic.size(), fp);
	std::rewind(fp);
	return n == kLfsMagic.size() && std::memcmp(buf.data(), kLfsMagic.data(), kLfsMagic.size()) == 0;
}

bool httpDownload(const std::string_view& url, const std::filesystem::path& dest, char* err, uint32_t err_cap) {
	std::filesystem::create_directories(dest.parent_path());

	std::ofstream ofs(dest, std::ios::binary);
	if (!ofs) {
		std::snprintf(
				err, err_cap, "httpDownload: cannot create '%s': %s", dest.string().c_str(), std::strerror(errno)
		);
		return false;
	}

	const cpr::Response r = cpr::Download(ofs, cpr::Url{url});
	ofs.close();

	if (r.error) {
		std::filesystem::remove(dest);
		std::snprintf(err, err_cap, "httpDownload: %s: %s", url.data(), r.error.message.c_str());
		return false;
	}
	if (r.status_code >= 400) {
		std::filesystem::remove(dest);
		std::snprintf(err, err_cap, "httpDownload: %s: HTTP %ld", url.data(), r.status_code);
		return false;
	}
	return true;
}

std::FILE*
lfsOpen(const std::string& git_url, const std::string& ref, const std::string& subpath,
		const std::filesystem::path& localRepo, char* err, uint32_t err_cap) {
	const auto cachePath = localRepo / ".tmm_lfs" / subpath;

	if (!std::filesystem::exists(cachePath)) {
		const std::string resolveUrl = git_url + "/resolve/" + ref + "/" + subpath;
		if (!httpDownload(resolveUrl, cachePath, err, err_cap)) {
			return nullptr;
		}
	}

	std::FILE* fp = std::fopen(cachePath.string().c_str(), "rb");
	if (fp == nullptr) {
		std::snprintf(err, err_cap, "lfsOpen: cannot open '%s': %s", cachePath.string().c_str(), std::strerror(errno));
	}
	return fp;
}

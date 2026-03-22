/**
 * @file format.hpp
 * @brief Compatibility shim for std::format (C++20/23).
 *
 * @details
 * Prefers the standard `<format>` header when the compiler provides it
 * (detected via the `__cpp_lib_format` feature-test macro).  Falls back
 * to fmtlib (`fmt::format`) on compilers that do not yet ship `<format>`.
 *
 * When the fallback is active, `fmt::format` and `fmt::print` are injected
 * into `namespace std` so that all existing `std::format` / `std::print`
 * usage compiles unchanged on both old and new compilers.
 *
 * @note Injecting names into `namespace std` is technically undefined
 *       behaviour per the C++ standard, but is widely accepted practice
 *       for polyfill headers and is handled correctly by every major
 *       compiler.  The injection is guarded behind the fallback path so
 *       it is never active when the real `<format>` is available.
 *
 * @see https://en.cppreference.com/w/cpp/utility/format
 * @see https://github.com/fmtlib/fmt
 */

#pragma once

#if defined(__cpp_lib_format) && __cpp_lib_format >= 202110L

#include <format>

#else // Fallback: fmtlib

#include <fmt/format.h>
#include <fmt/ostream.h>

// Inject fmt::format / fmt::print into namespace std so that all existing
// std::format / std::print code compiles without modification.
// NOLINTNEXTLINE(cert-dcl58-cpp) -- intentional polyfill injection; guarded by !__cpp_lib_format
namespace std {
	using ::fmt::format;
	using ::fmt::format_to;
	using ::fmt::format_to_n;
	using ::fmt::formatted_size;
	using ::fmt::make_format_args;
	using ::fmt::vformat;
} // namespace std

#endif // __cpp_lib_format

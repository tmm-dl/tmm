/**
 * @file expected.hpp
 * @brief Compatibility shim for std::expected (C++23).
 *
 * @details
 * Prefers the standard `<expected>` header when the compiler provides it
 * (detected via the `__cpp_lib_expected` feature-test macro).  Falls back
 * to the header-only TartanLlama/expected polyfill (`tl::expected`) on
 * compilers that do not yet ship C++23's `<expected>`.
 *
 * When the polyfill is active, `tl::expected`, `tl::unexpected`, and
 * `tl::bad_expected_access` are injected into `namespace std` so that
 * all existing `std::expected` / `std::unexpected` usage compiles
 * unchanged on both old and new compilers.
 *
 * @note Injecting names into `namespace std` is technically undefined
 *       behaviour per the C++ standard, but is widely accepted practice
 *       for polyfill headers and is handled correctly by every major
 *       compiler.  The injection is guarded behind the fallback path so
 *       it is never active when the real `<expected>` is available.
 *
 * @see https://en.cppreference.com/w/cpp/utility/expected
 * @see https://github.com/TartanLlama/expected
 */

#pragma once

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L

#	include <expected>

#else // Fallback: TartanLlama/expected

#	include <tl/expected.hpp>

// Inject tl::expected into namespace std so that all existing
// std::expected / std::unexpected code compiles without modification.
// NOLINTNEXTLINE(cert-dcl58-cpp) -- intentional polyfill injection; guarded by !__cpp_lib_expected
namespace std {
	template<class T, class E>
	using expected = ::tl::expected<T, E>;

	template<class E>
	using unexpected = ::tl::unexpected<E>;

	template<class E>
	using bad_expected_access = ::tl::bad_expected_access<E>;
} // namespace std

#endif // __cpp_lib_expected

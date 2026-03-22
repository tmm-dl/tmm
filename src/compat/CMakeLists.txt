# tmm_compat — header-only compatibility shims
#
# INTERFACE library: no compiled sources.  Consumers link against tmm_compat
# and get the include path plus any polyfill dependencies transitively.
add_library(tmm_compat INTERFACE)

target_sources(tmm_compat
    INTERFACE
        FILE_SET tmm_compat_headers
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_LIST_DIR}/../../include
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/compat/expected.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/compat/format.hpp
)

target_compile_features(tmm_compat INTERFACE cxx_std_23)

##########################################################################################
# Dependencies
##########################################################################################

# std::expected polyfill — used when the compiler does not provide <expected>.
# The shim (tmm/compat/expected.hpp) auto-detects via __cpp_lib_expected and
# falls back to tl::expected transparently; no call-site changes required.
CPMAddPackage(
    NAME expected
    GITHUB_REPOSITORY TartanLlama/expected
    GIT_TAG v1.1.0
    OPTIONS "EXPECTED_BUILD_TESTS OFF"
)
target_link_libraries(tmm_compat INTERFACE tl::expected)

# std::format polyfill — used when the compiler does not provide <format>.
# The shim (tmm/compat/format.hpp) auto-detects via __cpp_lib_format and
# falls back to fmtlib transparently; no call-site changes required.
CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 11.1.4
    OPTIONS "FMT_INSTALL OFF" "FMT_TEST OFF" "FMT_DOC OFF"
)
target_link_libraries(tmm_compat INTERFACE fmt::fmt-header-only)

# ttm_compat — header-only compatibility shims
#
# INTERFACE library: no compiled sources.  Consumers link against ttm_compat
# and get the include path plus any polyfill dependencies transitively.
add_library(ttm_compat INTERFACE)

target_sources(ttm_compat
    INTERFACE
        FILE_SET ttm_compat_headers
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_LIST_DIR}/../../include
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/compat/expected.hpp
)

target_compile_features(ttm_compat INTERFACE cxx_std_23)

##########################################################################################
# Dependencies
##########################################################################################

# std::expected polyfill — used when the compiler does not provide <expected>.
# The shim (ttm/compat/expected.hpp) auto-detects via __cpp_lib_expected and
# falls back to tl::expected transparently; no call-site changes required.
CPMAddPackage(
    NAME expected
    GITHUB_REPOSITORY TartanLlama/expected
    GIT_TAG v1.1.0
    OPTIONS "EXPECTED_BUILD_TESTS OFF"
)
target_link_libraries(ttm_compat INTERFACE tl::expected)

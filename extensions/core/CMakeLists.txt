cmake_minimum_required(VERSION 3.24)
project(ttm-core-plugin CXX)

add_library(ttm_core MODULE src/core.cpp)

target_compile_features(ttm_core PRIVATE cxx_std_23)

target_include_directories(ttm_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(ttm_core PROPERTIES
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# ---------------------------------------------------------------------------
# libgit2 — reuse the instance already fetched by ttm_datasets when building
# from the root.  When building standalone, require an installed libgit2.
# ---------------------------------------------------------------------------
if(NOT TARGET libgit2)
    find_package(libgit2 QUIET)
    if(NOT libgit2_FOUND)
        message(FATAL_ERROR
            "ttm-core-plugin requires libgit2.  "
            "Build from the repository root with -DTTM_BUILD_EXTENSIONS=ON, "
            "or install libgit2 and re-run cmake.")
    endif()
endif()

target_link_libraries(ttm_core PRIVATE libgit2 libgit2package)

install(TARGETS ttm_core DESTINATION .)

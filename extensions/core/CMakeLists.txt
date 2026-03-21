cmake_minimum_required(VERSION 3.24)
project(ttm-core-plugin CXX)

add_library(ttm_core MODULE
    src/plugin.cpp
    src/git_source.cpp
    src/netutils.cpp
    src/tvm_loader.cpp
    src/scheduler_state.cpp
    src/scheduler_constant.cpp
    src/scheduler_step.cpp
    src/scheduler_linear.cpp
    src/scheduler_cosine.cpp
    src/scheduler_cosine_warmup.cpp
    src/schedulers.cpp
)

target_compile_features(ttm_core PRIVATE cxx_std_23)

target_include_directories(ttm_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_BINARY_DIR}
)

# dlpack — abi.h includes <dlpack/dlpack.h>; CPM deduplicates when built from root.
CPMAddPackage(URI "gh:dmlc/dlpack@0.8" EXCLUDE_FROM_ALL YES)
target_link_libraries(ttm_core PRIVATE dlpack)

set_target_properties(ttm_core PROPERTIES
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

include(GenerateExportHeader)
generate_export_header(ttm_core
    EXPORT_FILE_NAME ttm_core_export.h
    EXPORT_MACRO_NAME TTM_CORE_EXPORT
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

# cpr — C++ HTTP client used to download git-lfs objects that libgit2 skips.
CPMAddPackage(URI "gh:libcpr/cpr#1.14.1" OPTIONS "CPR_CURL_USE_LIBPSL OFF" EXCLUDE_FROM_ALL YES)
target_link_libraries(ttm_core PRIVATE cpr::cpr)

install(TARGETS ttm_core DESTINATION .)

# ---------------------------------------------------------------------------
# Tests — add when building tests are enabled (mirrors root TMM_BUILD_TESTS).
# Guarded so standalone `cmake -S extensions/core` still works.
# ---------------------------------------------------------------------------
if(TMM_BUILD_TESTS OR (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME))
    include(CTest)
    add_subdirectory(tests)
endif()

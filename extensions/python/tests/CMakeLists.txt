cmake_minimum_required(VERSION 3.24)
project(tmm-python-tests CXX)

# ---------------------------------------------------------------------------
# Catch2 — deduplicates when built from the root (CPM caches by package name)
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME Catch2
    GITHUB_REPOSITORY catchorg/Catch2
    VERSION 3.8.0
    EXCLUDE_FROM_ALL YES
)

# ---------------------------------------------------------------------------
# dlpack — needed by abi.h for DLTensor
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME dlpack
    GITHUB_REPOSITORY dmlc/dlpack
    VERSION 0.8
    EXCLUDE_FROM_ALL YES
)

# ---------------------------------------------------------------------------
# tmm_python_tests executable
# ---------------------------------------------------------------------------
add_executable(tmm_python_tests
    test_optimizer_vtables.cpp
)

target_compile_features(tmm_python_tests PRIVATE cxx_std_23)

target_include_directories(tmm_python_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

target_link_libraries(tmm_python_tests PRIVATE
    Catch2::Catch2WithMain
    dlpack
    ${CMAKE_DL_LIBS}
)

# Inject the path of the built tmm_python module so tests can load it via dlopen.
target_compile_definitions(tmm_python_tests PRIVATE
    TMM_PYTHON_PLUGIN_PATH="$<TARGET_FILE:tmm_python>"
)

# tmm_python must be built before the tests run.
add_dependencies(tmm_python_tests tmm_python)

# ---------------------------------------------------------------------------
# CTest integration — register Catch2 test cases so `ctest` picks them up.
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(tmm_python_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

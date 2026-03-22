cmake_minimum_required(VERSION 3.24)
project(tmm-python-plugin CXX)

##########################################################################################
# Dependencies
##########################################################################################

# Python 3 with embedding support
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# PyTorch — optional but required for the full PyTorch model loader.
# If PyTorch is not installed, the plugin still compiles but returns an error at load time.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_CMAKE_PREFIX)
    find_package(Torch QUIET HINTS "${TORCH_CMAKE_PREFIX}")
endif()

##########################################################################################
# tmm_python shared library
##########################################################################################

add_library(tmm_python MODULE
    src/python_model.cpp
    src/python_optimizer.cpp
    src/python_transform.cpp
)

target_compile_features(tmm_python PRIVATE cxx_std_23)

target_include_directories(tmm_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_BINARY_DIR}
)

# dlpack — abi.h includes <dlpack/dlpack.h>; CPM deduplicates when built from root.
CPMAddPackage(
    NAME dlpack
    GITHUB_REPOSITORY dmlc/dlpack
    VERSION 0.8
    EXCLUDE_FROM_ALL YES
)
target_link_libraries(tmm_python PRIVATE dlpack)

set_target_properties(tmm_python PROPERTIES
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

include(GenerateExportHeader)
generate_export_header(tmm_python
    EXPORT_FILE_NAME tmm_python_export.h
    EXPORT_MACRO_NAME TMM_PYTHON_EXPORT
)

target_link_libraries(tmm_python PRIVATE Python3::Python)

if(Torch_FOUND)
    target_link_libraries(tmm_python PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(tmm_python PRIVATE TMM_PYTHON_HAS_TORCH)

    message(STATUS "tmm-python-plugin: PyTorch found — full loader enabled")
else()
    message(STATUS "tmm-python-plugin: PyTorch not found — stub loader only")
endif()

include(GNUInstallDirs)
install(TARGETS tmm_python LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}")

# ---------------------------------------------------------------------------
# Tests — add when building tests are enabled (mirrors root TMM_BUILD_TESTS).
# Guarded so standalone `cmake -S extensions/python` still works.
# ---------------------------------------------------------------------------
if(TMM_BUILD_TESTS OR (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME))
    include(CTest)
    add_subdirectory(tests)
endif()

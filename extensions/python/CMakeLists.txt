cmake_minimum_required(VERSION 3.24)
project(ttm-python-plugin CXX)

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
# ttm_python shared library
##########################################################################################

add_library(ttm_python MODULE src/python_model.cpp)

target_compile_features(ttm_python PRIVATE cxx_std_23)

target_include_directories(ttm_python PRIVATE
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
target_link_libraries(ttm_python PRIVATE dlpack)

set_target_properties(ttm_python PROPERTIES
    PREFIX ""
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

include(GenerateExportHeader)
generate_export_header(ttm_python
    EXPORT_FILE_NAME ttm_python_export.h
    EXPORT_MACRO_NAME TTM_PYTHON_EXPORT
)

target_link_libraries(ttm_python PRIVATE Python3::Python)

if(Torch_FOUND)
    target_link_libraries(ttm_python PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(ttm_python PRIVATE TTM_PYTHON_HAS_TORCH)
    message(STATUS "ttm-python-plugin: PyTorch found — full loader enabled")
else()
    message(STATUS "ttm-python-plugin: PyTorch not found — stub loader only")
endif()

install(TARGETS ttm_python DESTINATION .)

##########################################################################################
# Tests — ttm_tests executable (Catch2)
##########################################################################################
include(CTest)

# Catch2 via CPM (consistent with the rest of the project)
CPMAddPackage(URI "gh:catchorg/Catch2@3.8.0" EXCLUDE_FROM_ALL YES)

add_executable(ttm_tests
    plugins/test_byte_reader.cpp
    plugins/test_plugin_manager.cpp
    datasets/test_dataset_info.cpp
    model/test_device.cpp
    model/test_collator.cpp
    model/test_model_loader.cpp
    model/test_python_plugin_integration.cpp
)

target_compile_features(ttm_tests PRIVATE cxx_std_23)

target_link_libraries(ttm_tests PRIVATE
    ttm_model
    ttm_plugins
    ttm_datasets
    Catch2::Catch2WithMain
)

# When the Python extension is built, inject its output path so the integration
# test can locate ttm_python.so without requiring a manual env var.
if(TARGET ttm_python)
    target_compile_definitions(ttm_tests PRIVATE
        TTM_PYTHON_PLUGIN_PATH="$<TARGET_FILE:ttm_python>"
    )
    add_dependencies(ttm_tests ttm_python)
endif()

# Discover individual test cases so ctest can filter / run them in parallel.
# Catch2 ships its CMake helpers under extras/ when fetched via CPM.
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(ttm_tests)

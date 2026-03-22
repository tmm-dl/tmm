##########################################################################################
# Tests — tmm_tests executable (Catch2)
##########################################################################################
include(CTest)

# Catch2 via CPM (consistent with the rest of the project)
CPMAddPackage(URI "gh:catchorg/Catch2@3.8.0" EXCLUDE_FROM_ALL YES)

add_executable(tmm_tests
    conf/test_config_loader.cpp
    trainer/test_trainer_loop.cpp
    plugins/test_byte_reader.cpp
    plugins/test_plugin_manager.cpp
    datasets/test_dataset_info.cpp
    model/test_device.cpp
    model/test_collator.cpp
    model/test_model_loader.cpp
    model/test_python_plugin_integration.cpp
)

target_compile_features(tmm_tests PRIVATE cxx_std_23)

target_link_libraries(tmm_tests PRIVATE
    tmm_conf
    tmm_trainer
    tmm_model
    tmm_plugins
    tmm_datasets
    Catch2::Catch2WithMain
)

# When the Python extension is built, inject its output path so the integration
# test can locate tmm_python.so without requiring a manual env var.
if(TARGET tmm_python)
    target_compile_definitions(tmm_tests PRIVATE
        TMM_PYTHON_PLUGIN_PATH="$<TARGET_FILE:tmm_python>"
    )
    add_dependencies(tmm_tests tmm_python)
endif()

# Discover individual test cases so ctest can filter / run them in parallel.
# Catch2 ships its CMake helpers under extras/ when fetched via CPM.
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(tmm_tests)

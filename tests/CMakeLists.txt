##########################################################################################
# Tests — ttm_tests executable (Catch2)
##########################################################################################
include(CTest)

# Catch2 via CPM (consistent with the rest of the project)
CPMAddPackage(URI "gh:catchorg/Catch2@3.8.0" EXCLUDE_FROM_ALL YES)

add_executable(ttm_tests
    plugins/test_byte_reader.cpp
    plugins/test_plugin_manager.cpp
)

target_compile_features(ttm_tests PRIVATE cxx_std_20)

target_link_libraries(ttm_tests PRIVATE
    ttm_plugins
    Catch2::Catch2WithMain
)

# Discover individual test cases so ctest can filter / run them in parallel.
# Catch2 ships its CMake helpers under extras/ when fetched via CPM.
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(ttm_tests)

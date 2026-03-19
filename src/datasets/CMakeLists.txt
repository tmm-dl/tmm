add_library(ttm_datasets)

target_sources(ttm_datasets
    PUBLIC
        FILE_SET ttm_datasets_headers
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_LIST_DIR}/../../include
    PRIVATE
        datasets.cpp
        dataset_info.cpp
        dataset_loader.cpp
)

target_compile_features(ttm_datasets PRIVATE cxx_std_23)
target_include_directories(ttm_datasets PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)
target_link_libraries(ttm_datasets PUBLIC ttm_plugins)

##########################################################################################
# Dependencies
##########################################################################################
# LibGit2 — fetched here so extensions/core can reuse the same CPM instance.
# ttm_datasets itself does notuse libgit2; only extensions/core does.
CPMAddPackage(
	URI "gh:libgit2/libgit2@1.8.4"
	OPTIONS "USE_SSH OFF" "USE_HTTPS ON" "REGEX_BACKEND builtin" "USE_BUNDLED_ZLIB OFF" "BUILD_CLI OFF" "BUILD_TESTS OFF" "DEPRECATE_HARD OFF"
	EXCLUDE_FROM_ALL YES
)

# Apache Arrow
# DOWNLOAD_ONLY so we can fix arrow_SOURCE_DIR before add_subdirectory.
# Arrow's ThirdpartyToolchain.cmake resolves thirdparty/ relative to
# arrow_SOURCE_DIR, but FetchContent sets that to the archive root while
# the actual thirdparty/ lives under cpp/.
CPMAddPackage(
	NAME arrow
	URL "https://github.com/apache/arrow/archive/refs/tags/apache-arrow-19.0.1.tar.gz"
	DOWNLOAD_ONLY YES
)
if(arrow_SOURCE_DIR)
	# Override as a normal variable so add_subdirectory inherits root/cpp, and
	# Arrow's ThirdpartyToolchain.cmake resolves ${arrow_SOURCE_DIR}/thirdparty
	# to cpp/thirdparty where versions.txt actually lives.
	set(arrow_SOURCE_DIR "${arrow_SOURCE_DIR}/cpp")
	set(ARROW_BUILD_STATIC ON CACHE BOOL "" FORCE)
	set(ARROW_BUILD_SHARED OFF CACHE BOOL "" FORCE)
	set(ARROW_DEPENDENCY_SOURCE "BUNDLED" CACHE STRING "" FORCE)
	set(ARROW_SIMD_LEVEL "NONE" CACHE STRING "" FORCE)
	set(ARROW_RUNTIME_SIMD_LEVEL "NONE" CACHE STRING "" FORCE)
	set(ARROW_WITH_RE2 OFF CACHE BOOL "" FORCE)
	set(ARROW_WITH_UTF8PROC OFF CACHE BOOL "" FORCE)
	set(ARROW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(ARROW_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
	set(ARROW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	set(ARROW_PARQUET ON CACHE BOOL "" FORCE)
	set(ARROW_IPC ON CACHE BOOL "" FORCE)
	set(ARROW_WITH_SNAPPY ON CACHE BOOL "" FORCE)
	# Arrow static libs must be PIC because ttm_plugins is a shared library
	set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
	add_subdirectory("${arrow_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/arrow-build" EXCLUDE_FROM_ALL)
	# Arrow uses include_directories() (not target_include_directories), so its
	# INTERFACE_INCLUDE_DIRECTORIES are not propagated to consumers.  Add them explicitly.
	target_include_directories(ttm_datasets PRIVATE
		"${arrow_SOURCE_DIR}/src"
		"${CMAKE_BINARY_DIR}/_deps/arrow-build/src")
endif()
target_link_libraries(ttm_datasets PRIVATE arrow_static parquet_static)

# yaml-cpp — for HuggingFace Dataset Card frontmatter parsing
CPMAddPackage(
	URI "gh:jbeder/yaml-cpp#yaml-cpp-0.9.0"
	OPTIONS "YAML_CPP_BUILD_TESTS OFF" "YAML_CPP_BUILD_TOOLS OFF" "YAML_CPP_INSTALL OFF"
	EXCLUDE_FROM_ALL YES
)
target_link_libraries(ttm_datasets PRIVATE yaml-cpp::yaml-cpp)


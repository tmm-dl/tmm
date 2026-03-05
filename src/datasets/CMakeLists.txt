add_library(ttm_datasets)

target_sources(ttm_datasets
    PUBLIC
        FILE_SET ttm_datasets_headers
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/datasets
    PRIVATE
        datasets.cpp
)

##########################################################################################
# Dependencies
##########################################################################################
# LibGit
CPMAddPackage(
	URI "gh:libgit2/libgit2@1.8.4"
	OPTIONS "USE_SSH OFF" "USE_HTTPS OFF" "REGEX_BACKEND builtin" "USE_BUNDLED_ZLIB OFF" "BUILD_CLI OFF" "BUILD_TESTS OFF" "DEPRECATE_HARD OFF"
	EXCLUDE_FROM_ALL YES
)
target_link_libraries(ttm_datasets PRIVATE libgit2 libgit2package)

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
	add_subdirectory("${arrow_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/arrow-build" EXCLUDE_FROM_ALL)
endif()
target_link_libraries(ttm_datasets PRIVATE arrow_static)


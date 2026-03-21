cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

##########################################################################################
# Package Features
##########################################################################################
include(FeatureSummary)
option(TMM_BUILD_TESTS "Build tests" ON)
option(TTM_BUILD_EXTENSIONS "Build plugin extensions (extensions/core, extensions/nlp-tasks)" OFF)
set(TTM_WASM_TOOLCHAIN "" CACHE FILEPATH
    "Path to the WASI SDK CMake toolchain file (required for WASM extensions)")
# Auto-fetch WASI SDK via FetchContent when building WASM extensions and no
# toolchain was supplied explicitly.  The tarball is cached in CPM_SOURCE_CACHE
# (or the build tree) and never re-downloaded on subsequent configures.
if(TTM_BUILD_EXTENSIONS AND NOT TTM_WASM_TOOLCHAIN)
    set(_wasi_version "32")
    set(_wasi_base "wasi-sdk-${_wasi_version}.0")
    # Resolve the platform-specific asset name
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(_wasi_asset "${_wasi_base}-arm64-macos")
        else()
            set(_wasi_asset "${_wasi_base}-x86_64-macos")
        endif()
    else()  # Linux (and other Unix)
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(_wasi_asset "${_wasi_base}-arm64-linux")
        else()
            set(_wasi_asset "${_wasi_base}-x86_64-linux")
        endif()
    endif()

    include(FetchContent)
    FetchContent_Declare(wasi_sdk
        URL "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-${_wasi_version}/${_wasi_asset}.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP YES
    )
    FetchContent_GetProperties(wasi_sdk)
    if(NOT wasi_sdk_POPULATED)
        message(STATUS "ttm: Fetching WASI SDK ${_wasi_base} (${_wasi_asset})...")
        FetchContent_Populate(wasi_sdk)
    endif()
    set(TTM_WASM_TOOLCHAIN "${wasi_sdk_SOURCE_DIR}/share/cmake/wasi-sdk.cmake" CACHE FILEPATH
        "Path to the WASI SDK CMake toolchain file (required for WASM extensions)" FORCE)
    message(STATUS "ttm: WASI SDK toolchain: ${TTM_WASM_TOOLCHAIN}")
    unset(_wasi_version)
    unset(_wasi_base)
    unset(_wasi_asset)
endif()

add_feature_info(tests TMM_BUILD_TESTS "Build tests")
feature_summary(WHAT ALL)

##########################################################################################
# Versioning
##########################################################################################
find_package(Git)

if(GIT_EXECUTABLE)
	execute_process(
		COMMAND ${GIT_EXECUTABLE} describe --tags --dirty
		WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
		OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
		RESULT_VARIABLE GIT_DESCRIBE_ERROR_CODE
		OUTPUT_STRIP_TRAILING_WHITESPACE
    )
	if(NOT GIT_DESCRIBE_ERROR_CODE)
		string(REGEX MATCH "([0-9]+).([0-9]+).([0-9]+)(-.+)?" _ "${GIT_DESCRIBE_VERSION}")
		set(TMM_VER_MAJOR ${CMAKE_MATCH_1} CACHE INTERNAL "")
		set(TMM_VER_MINOR ${CMAKE_MATCH_2} CACHE INTERNAL "")
		set(TMM_VER_PATCH ${CMAKE_MATCH_3} CACHE INTERNAL "")
		set(TMM_VER_META  ${CMAKE_MATCH_4} CACHE INTERNAL "")
	else()
		message(WARNING "Could not extract version information from the latest tag, ${GIT_DESCRIBE_VERSION}")
	endif()
endif()
if(NOT DEFINED TMM_VER_MAJOR OR NOT DEFINED TMM_VER_MINOR OR NOT DEFINED TMM_VER_PATCH)
	message(WARNING "I failed to read the current version from the latest git tag.")

	set(TMM_VER_MAJOR "0" CACHE INTERNAL "")
	set(TMM_VER_MINOR "0" CACHE INTERNAL "")
	set(TMM_VER_PATCH "0" CACHE INTERNAL "")
	set(TMM_VER_META  "-dev" CACHE INTERNAL "")  # https://semver.org/#spec-item-10
endif()

# Does not include the meta component since this will be used in places where CMake expects the form X.X.X
set(TMM_VERSION "${TMM_VER_MAJOR}.${TMM_VER_MINOR}.${TMM_VER_PATCH}")

##########################################################################################
# Project Setup
##########################################################################################
project(tmm VERSION ${TMM_VERSION} LANGUAGES CXX)

# Load CPM
set(CPM_DOWNLOAD_VERSION 0.42.0)
set(CPM_HASH_SUM "2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a")

if(CPM_SOURCE_CACHE)
set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

file(DOWNLOAD
	https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
	${CPM_DOWNLOAD_LOCATION} EXPECTED_HASH SHA256=${CPM_HASH_SUM}
	TLS_VERIFY OFF # Sometimes broken in the GitHub Action; we have the checksum to verify that there is nothing malicious going on; try at some point to remove this line
)

include(${CPM_DOWNLOAD_LOCATION})
message("Loaded CPM v${CURRENT_CPM_VERSION}")

# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_subdirectory(src)

########################################################################################################################
# Plugin Extensions                                                                                                    #
########################################################################################################################
if(TTM_BUILD_EXTENSIONS)
    # Native core plugin — shares the libgit2 instance fetched by ttm_datasets.
    # Must be added after src/ so that the libgit2 CPM package is already available.
    add_subdirectory(extensions/core)

    # Python model loader — requires Python3; PyTorch optional (stub only without it).
    find_package(Python3 QUIET COMPONENTS Interpreter Development)
    if(Python3_FOUND)
        add_subdirectory(extensions/python)
    else()
        message(STATUS "ttm: Python3 not found — skipping extensions/python")
    endif()

    # console-ui native plugin — full FTXUI component API (ScreenInteractive,
    # pthreads) for the optional "interactive":true mode.  The resolver prefers
    # .so over .wasm, so this automatically supersedes the WASM build when both
    # are present.
    CPMAddPackage(URI "gh:ArthurSonzogni/FTXUI@6.1.9"
        NAME FTXUI
        OPTIONS
            "FTXUI_BUILD_EXAMPLES OFF"
            "FTXUI_BUILD_TESTS    OFF"
            "FTXUI_BUILD_DOCS     OFF"
            "FTXUI_ENABLE_INSTALL OFF"
        EXCLUDE_FROM_ALL YES
    )
    add_library(ttm_console_ui MODULE
        extensions/console-ui/src/console_ui.cpp
    )
    set_target_properties(ttm_console_ui PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extensions/console-ui"
    )
    target_compile_features(ttm_console_ui PRIVATE cxx_std_20)
    target_include_directories(ttm_console_ui PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    # dlpack is already fetched by extensions/core above; CPM deduplicates.
    CPMAddPackage(URI "gh:dmlc/dlpack@0.8" EXCLUDE_FROM_ALL YES)
    target_link_libraries(ttm_console_ui PRIVATE
        ftxui::screen
        ftxui::dom
        ftxui::component
        dlpack
    )

    # WASM plugins require WASI SDK; cross-compiled via ExternalProject so they
    # do not pollute the host build tree.
    if(TTM_WASM_TOOLCHAIN)
        include(ExternalProject)

        ExternalProject_Add(ttm_nlp_tasks_wasm
            SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/extensions/nlp-tasks
            BINARY_DIR  ${CMAKE_BINARY_DIR}/extensions/nlp-tasks
            CMAKE_ARGS
                -DCMAKE_TOOLCHAIN_FILE=${TTM_WASM_TOOLCHAIN}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extensions
            BUILD_ALWAYS YES
            INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
        )

        # console-ui: live TUI dashboard plugin (FTXUI DOM, WASI).
        # FTXUI is fetched by the sub-project via CPM; compiled with
        # -D__EMSCRIPTEN__=1 so Terminal::Size() skips ioctl.
        ExternalProject_Add(ttm_console_ui_wasm
            SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/extensions/console-ui
            BINARY_DIR  ${CMAKE_BINARY_DIR}/extensions/console-ui
            CMAKE_ARGS
                -DCMAKE_TOOLCHAIN_FILE=${TTM_WASM_TOOLCHAIN}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/extensions
                -DCPM_SOURCE_CACHE=${CPM_SOURCE_CACHE}
            BUILD_ALWAYS YES
            INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR>
        )
    else()
        message(STATUS
            "TTM_WASM_TOOLCHAIN not set; skipping WASM plugins (nlp-tasks, console-ui). "
            "Set -DTTM_WASM_TOOLCHAIN=/opt/wasi-sdk/share/cmake/wasi-sdk.cmake to enable.")
    endif()
endif()

########################################################################################################################
# Coverage                                                                                                             #
########################################################################################################################
option(TTM_ENABLE_COVERAGE "Enable test coverage instrumentation (GCC / Clang only)" OFF)
if(TTM_ENABLE_COVERAGE)
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		add_compile_options(--coverage -O0 -g)
		add_link_options(--coverage)
	else()
		message(WARNING "TTM_ENABLE_COVERAGE is only supported with GCC/Clang; ignoring.")
	endif()
endif()

########################################################################################################################
# Testing                                                                                                              #
########################################################################################################################
# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING) AND TMM_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()
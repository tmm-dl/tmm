cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(ttm-console-ui-plugin CXX)

# ---------------------------------------------------------------------------
# Compiler requirements
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# CPM — lightweight package manager
# ---------------------------------------------------------------------------
if(NOT DEFINED CPM_DOWNLOAD_VERSION)
    set(CPM_DOWNLOAD_VERSION 0.42.0)
endif()
set(_cpm_file "${CMAKE_BINARY_DIR}/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS "${_cpm_file}")
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        "${_cpm_file}"
    )
endif()
include("${_cpm_file}")

# ---------------------------------------------------------------------------
# dlpack — abi.h transitively includes <dlpack/dlpack.h>.  This ExternalProject
# has its own build tree, so the parent's CPM dlpack instance is not visible.
# ---------------------------------------------------------------------------
CPMAddPackage(URI "gh:dmlc/dlpack@0.8" EXCLUDE_FROM_ALL YES)

# ---------------------------------------------------------------------------
# FTXUI — DOM-only (ftxui_screen + ftxui_dom).
# Component is excluded because it uses pthreads / POSIX signals which are
# unavailable in WASI.
#
# __EMSCRIPTEN__ is set as a global definition (not PRIVATE to the plugin)
# so FTXUI's own sources also skip the ioctl/winsize path in terminal.cpp.
# ---------------------------------------------------------------------------
add_compile_definitions(__EMSCRIPTEN__)
add_compile_options(-fno-exceptions)  # WASI has no C++ exception runtime

CPMAddPackage(URI "gh:ArthurSonzogni/FTXUI@6.1.9"
    OPTIONS
        "FTXUI_BUILD_EXAMPLES OFF"
        "FTXUI_BUILD_TESTS OFF"
        "FTXUI_BUILD_DOCS OFF"
        "FTXUI_ENABLE_INSTALL OFF"
    EXCLUDE_FROM_ALL YES
)

# ---------------------------------------------------------------------------
# Plugin WASM module
# Using add_executable rather than MODULE because CMake's MODULE type passes
# -shared to the linker, which wasm-ld rejects when --export-table is also
# present.  add_executable + -Wl,--no-entry produces a proper standalone WASM
# module without a _start entry point.
# ---------------------------------------------------------------------------
add_executable(ttm_console_ui src/console_ui.cpp)

target_include_directories(ttm_console_ui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(ttm_console_ui PRIVATE
    ftxui::screen
    ftxui::dom
    dlpack
)

# WASM/WASI-specific settings
target_compile_options(ttm_console_ui PRIVATE
    -Wall
    -Wextra
)

set_target_properties(ttm_console_ui PROPERTIES
    PREFIX    ""
    SUFFIX    ".wasm"
    # Hide all symbols by default — only export the ABI entry points.
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# WASM linker flags: no-entry (WASI does not call _start), export-table so
# the host can call back into the plugin's function table, and explicit ABI
# export list.
target_link_options(ttm_console_ui PRIVATE
    -Wl,--no-entry
    -Wl,--export-table
    -Wl,--export=ttm_plugin_get_info
    -Wl,--export=ttm_plugin_init
    -Wl,--export=ttm_plugin_teardown
    -Wl,--export=ttm_on_fit_begin
    -Wl,--export=ttm_on_epoch_begin
    -Wl,--export=ttm_on_batch_end
    -Wl,--export=ttm_on_epoch_end
    -Wl,--export=ttm_on_fit_end
    -Wl,--export=ttm_on_log
    -Wl,--export=ttm_on_metric
    -Wl,--export=__heap_base
    -Wl,--export=__data_end
    -Wl,--export=malloc
    -Wl,--export=free
)

# ---------------------------------------------------------------------------
# Install the .wasm into the extensions output directory
# ---------------------------------------------------------------------------
install(TARGETS ttm_console_ui DESTINATION .)

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
# FTXUI — DOM-only (ftxui_screen + ftxui_dom).
# Component is excluded because it uses pthreads / POSIX signals which are
# unavailable in WASI.
# ---------------------------------------------------------------------------
CPMAddPackage(
    NAME FTXUI
    GITHUB_REPOSITORY ArthurSonzogni/FTXUI
    GIT_TAG v0.11.1
    OPTIONS
        "FTXUI_BUILD_EXAMPLES OFF"
        "FTXUI_BUILD_TESTS OFF"
        "FTXUI_BUILD_DOCS OFF"
        "FTXUI_ENABLE_INSTALL OFF"
)

# ---------------------------------------------------------------------------
# Plugin shared library / WASM module
# ---------------------------------------------------------------------------
add_library(ttm_console_ui MODULE src/console_ui.cpp)

target_include_directories(ttm_console_ui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(ttm_console_ui PRIVATE
    ftxui::screen
    ftxui::dom
)

# WASM/WASI-specific settings
target_compile_options(ttm_console_ui PRIVATE
    -Wall
    -Wextra
    # Prevent FTXUI's Terminal::Size() from calling ioctl (unavailable in WASI).
    # With this flag active, FTXUI falls back to a static {80,24} default;
    # the plugin always passes an explicit Dimension::Fixed(w,h) anyway.
    -D__EMSCRIPTEN__=1
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
)

# ---------------------------------------------------------------------------
# Install the .wasm into the extensions output directory
# ---------------------------------------------------------------------------
install(TARGETS ttm_console_ui DESTINATION .)

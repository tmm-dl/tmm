cmake_minimum_required(VERSION 3.24)
project(ttm-nlp-tasks-plugin C)

add_library(ttm_nlp_tasks MODULE src/nlp_tasks.c)

target_compile_features(ttm_nlp_tasks PRIVATE c_std_11)
target_compile_options(ttm_nlp_tasks PRIVATE -Wall -Wextra)

target_include_directories(ttm_nlp_tasks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(ttm_nlp_tasks PROPERTIES
    PREFIX ""
    SUFFIX ".wasm"
    LINK_FLAGS
        "-Wl,--no-entry \
         -Wl,--export-table \
         -Wl,--export=ttm_plugin_get_info \
         -Wl,--export=ttm_plugin_init \
         -Wl,--export=ttm_plugin_teardown \
         -Wl,--export=__heap_base \
         -Wl,--export=__data_end"
)

install(TARGETS ttm_nlp_tasks DESTINATION .)

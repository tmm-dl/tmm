cmake_minimum_required(VERSION 3.24)
project(tmm-nlp-tasks-plugin C)

add_executable(tmm_nlp_tasks src/nlp_tasks.c)

target_compile_features(tmm_nlp_tasks PRIVATE c_std_11)
target_compile_options(tmm_nlp_tasks PRIVATE -Wall -Wextra)

target_include_directories(tmm_nlp_tasks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(DLPACK_INCLUDE_DIR)
    target_include_directories(tmm_nlp_tasks PRIVATE ${DLPACK_INCLUDE_DIR})
endif()

set_target_properties(tmm_nlp_tasks PROPERTIES
    SUFFIX ".wasm"
    LINK_FLAGS
        "-Wl,--no-entry \
         -Wl,--export-table \
         -Wl,--export=tmm_plugin_get_info \
         -Wl,--export=tmm_plugin_init \
         -Wl,--export=tmm_plugin_teardown \
         -Wl,--export=__heap_base \
         -Wl,--export=__data_end"
)

install(TARGETS tmm_nlp_tasks DESTINATION .)

add_library(ttm_plugins)

target_sources(ttm_plugins
    PUBLIC
        FILE_SET ttm_plugins_headers
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/plugins
    PRIVATE
        plugins.cpp
)

##########################################################################################
# Dependencies
##########################################################################################

# WAMR
CPMAddPackage(URI "gh:bytecodealliance/wasm-micro-runtime#WAMR-2.4.4" EXCLUDE_FROM_ALL YES)
target_link_libraries(ttm_plugins PRIVATE wamr)
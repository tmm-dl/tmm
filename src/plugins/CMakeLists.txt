add_library(tmm_plugins)

target_sources(tmm_plugins
    PUBLIC
        FILE_SET tmm_plugins_headers
        TYPE HEADERS
        BASE_DIRS
            ${CMAKE_CURRENT_LIST_DIR}/../../include
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/plugins/abi.h
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/plugins/extension.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/plugins/plugin_manager.hpp
    PRIVATE
        plugin_manager.cpp
        wasm_loader.cpp
        native_loader.cpp
        byte_reader.cpp
)

target_compile_features(tmm_plugins PUBLIC cxx_std_23)

# FILE_SET HEADERS propagates the BASE_DIRS as INTERFACE include paths for
# consumers, but not for the library's own private compilation units.
target_include_directories(tmm_plugins PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)

##########################################################################################
# Dependencies
##########################################################################################

# WAMR
CPMAddPackage(URI "gh:bytecodealliance/wasm-micro-runtime#WAMR-2.4.4" EXCLUDE_FROM_ALL YES)
target_link_libraries(tmm_plugins PRIVATE vmlib)
# WAMR does not set INTERFACE_INCLUDE_DIRECTORIES on its target; add manually.
if(wasm-micro-runtime_SOURCE_DIR)
    target_include_directories(tmm_plugins PRIVATE
        "${wasm-micro-runtime_SOURCE_DIR}/core/iwasm/include"
    )
endif()

target_link_libraries(tmm_plugins PUBLIC tmm_compat)

# dlopen / LoadLibrary for native plugin loading
target_link_libraries(tmm_plugins PRIVATE ${CMAKE_DL_LIBS})

# DLPack — header-only tensor ABI shared with the model system.
# Linked PUBLIC so consumers of tmm_plugins (including tmm_model) get the
# dlpack include path without having to add it themselves.
CPMAddPackage(
    NAME dlpack
    GITHUB_REPOSITORY dmlc/dlpack
    VERSION 0.8
    EXCLUDE_FROM_ALL YES
)
target_link_libraries(tmm_plugins PUBLIC dlpack)

# Arrow headers needed by plugin_manager.hpp → interfaces.hpp.
# Propagated PUBLIC so consumers of tmm_plugins also find them.
if(arrow_SOURCE_DIR)
    target_include_directories(tmm_plugins PUBLIC
        "${arrow_SOURCE_DIR}/src"
        "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    )
endif()
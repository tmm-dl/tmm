add_library(ttm_conf)

target_sources(ttm_conf
    PUBLIC
        FILE_SET ttm_conf_headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../include
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/conf/config.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/conf/loader.hpp
    PRIVATE
        loader.cpp
)

target_compile_features(ttm_conf PUBLIC cxx_std_23)
target_include_directories(ttm_conf PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)

# TTM_VERSION compile definition — propagated to all consumers (trainer, executable, …)
target_compile_definitions(ttm_conf PUBLIC TTM_VERSION="${TMM_VERSION}")

target_link_libraries(ttm_conf PUBLIC ttm_compat)

# yaml-cpp — CPM deduplicates across ttm_datasets and ttm_conf
CPMAddPackage(
    URI "gh:jbeder/yaml-cpp#yaml-cpp-0.9.0"
    OPTIONS "YAML_CPP_BUILD_TESTS OFF" "YAML_CPP_BUILD_TOOLS OFF" "YAML_CPP_INSTALL OFF"
    EXCLUDE_FROM_ALL YES
)
target_link_libraries(ttm_conf PRIVATE yaml-cpp::yaml-cpp)

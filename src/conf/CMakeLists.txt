add_library(tmm_conf)

target_sources(tmm_conf
    PUBLIC
        FILE_SET tmm_conf_headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../include
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/conf/config.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/conf/loader.hpp
    PRIVATE
        loader.cpp
)

target_compile_features(tmm_conf PUBLIC cxx_std_23)
target_include_directories(tmm_conf PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)

# TMM_VERSION compile definition — propagated to all consumers (trainer, executable, …)
target_compile_definitions(tmm_conf PUBLIC TMM_VERSION="${TMM_VERSION}")

target_link_libraries(tmm_conf PUBLIC tmm_compat)

# yaml-cpp — CPM deduplicates across tmm_datasets and tmm_conf
CPMAddPackage(
    URI "gh:jbeder/yaml-cpp#yaml-cpp-0.9.0"
    OPTIONS "YAML_CPP_BUILD_TESTS OFF" "YAML_CPP_BUILD_TOOLS OFF" "YAML_CPP_INSTALL OFF"
    EXCLUDE_FROM_ALL YES
)
target_link_libraries(tmm_conf PRIVATE yaml-cpp::yaml-cpp)

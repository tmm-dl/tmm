add_executable(ttm
    main.cpp
)

target_compile_features(ttm PUBLIC cxx_std_23)
target_compile_definitions(ttm PUBLIC TTM_VERSION="${TTM_VERSION}")

##########################################################################################
# Dependencies
##########################################################################################
add_subdirectory(compat)
add_subdirectory(datasets)
add_subdirectory(plugins)
target_link_libraries(ttm PRIVATE ttm_datasets ttm_plugins)

# Arrow headers are PRIVATE to ttm_datasets; expose them to the main executable
# so main.cpp can use arrow::RecordBatch / arrow::Schema directly.
if(arrow_SOURCE_DIR)
	target_include_directories(ttm PRIVATE
		"${arrow_SOURCE_DIR}/src"
		"${CMAKE_BINARY_DIR}/_deps/arrow-build/src")
endif()

# spdlog
CPMAddPackage(
	URI "gh:gabime/spdlog@1.15.3"
	EXCLUDE_FROM_ALL YES
)
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)  # Is this needed?
target_link_libraries(ttm PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)

# CLI11
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)
target_link_libraries(ttm PRIVATE CLI11::CLI11)
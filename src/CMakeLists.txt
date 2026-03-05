add_executable(ttm
    main.cpp
)

target_compile_features(ttm PUBLIC cxx_std_20)
target_compile_definitions(ttm PUBLIC TTM_VERSION="${TTM_VERSION}")

##########################################################################################
# Dependencies
##########################################################################################
add_subdirectory(datasets)
add_subdirectory(plugins)
target_link_libraries(ttm PRIVATE ttm_datasets ttm_plugins)

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
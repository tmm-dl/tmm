##########################################################################################
# Library sub-targets (order matters for CPM deduplication)
##########################################################################################
add_subdirectory(compat)    # ttm_compat   — expected + format polyfills
add_subdirectory(plugins)   # ttm_plugins  — PluginManager, WAMR, native loader
add_subdirectory(datasets)  # ttm_datasets — Arrow/Parquet dataset reader
add_subdirectory(conf)      # ttm_conf     — YAML config loader
add_subdirectory(trainer)   # ttm_trainer  — Trainer + IModel/IOptimizer/ILRScheduler

##########################################################################################
# ttm executable
##########################################################################################
add_executable(ttm main.cpp)

# PRIVATE: these features / definitions are for the executable only.
target_compile_features(ttm PRIVATE cxx_std_23)
target_include_directories(ttm PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)

# ttm_trainer is the facade: it pulls in ttm_conf, ttm_datasets, ttm_plugins,
# Arrow headers, and TTM_VERSION transitively.
target_link_libraries(ttm PRIVATE ttm_trainer)

# CLI11 — argument parsing; linked only to the executable, never to any library.
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)
target_link_libraries(ttm PRIVATE CLI11::CLI11)

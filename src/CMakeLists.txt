##########################################################################################
# Library sub-targets (order matters for CPM deduplication)
##########################################################################################
add_subdirectory(compat)    # ttm_compat   — expected + format polyfills
add_subdirectory(plugins)   # ttm_plugins  — PluginManager, WAMR, native loader
add_subdirectory(datasets)  # ttm_datasets — Arrow/Parquet dataset reader
add_subdirectory(conf)      # ttm_conf     — YAML config loader
add_subdirectory(trainer)   # ttm_trainer  — Trainer + IModel/IOptimizer/ILRScheduler
add_subdirectory(model)     # ttm_model    — ModelPipeline, collator, device, params

##########################################################################################
# ttm executable
##########################################################################################
add_executable(ttm main.cpp)

# PRIVATE: these features / definitions are for the executable only.
target_compile_features(ttm PRIVATE cxx_std_23)
target_include_directories(ttm PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)

# ttm_model pulls in ttm_trainer (IModel/IOptimizer/Batch), ttm_plugins
# (IModelLoader, DLPack), and ttm_conf transitively.
target_link_libraries(ttm PRIVATE ttm_model)

# CLI11 — argument parsing; linked only to the executable, never to any library.
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)
target_link_libraries(ttm PRIVATE CLI11::CLI11)

##########################################################################################
# Library sub-targets (order matters for CPM deduplication)
##########################################################################################
add_subdirectory(compat)    # tmm_compat   — expected + format polyfills
add_subdirectory(plugins)   # tmm_plugins  — PluginManager, WAMR, native loader
add_subdirectory(datasets)  # tmm_datasets — Arrow/Parquet dataset reader
add_subdirectory(conf)      # tmm_conf     — YAML config loader
add_subdirectory(trainer)   # tmm_trainer  — Trainer + IModel/IOptimizer/ILRScheduler
add_subdirectory(model)     # tmm_model    — ModelPipeline, collator, device, params

##########################################################################################
# tmm executable
##########################################################################################
add_executable(tmm main.cpp)

# PRIVATE: these features / definitions are for the executable only.
target_compile_features(tmm PRIVATE cxx_std_23)
target_include_directories(tmm PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)

# tmm_model pulls in tmm_trainer (IModel/IOptimizer/Batch), tmm_plugins
# (IModelLoader, DLPack), and tmm_conf transitively.
target_link_libraries(tmm PRIVATE tmm_model)

# CLI11 — argument parsing; linked only to the executable, never to any library.
CPMAddPackage(URI "gh:CLIUtils/CLI11@2.4.2" EXCLUDE_FROM_ALL YES)
target_link_libraries(tmm PRIVATE CLI11::CLI11)

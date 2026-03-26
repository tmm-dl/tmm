add_library(tmm_trainer)

target_sources(tmm_trainer
    PUBLIC
        FILE_SET tmm_trainer_headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../include
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/trainer/callback.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/trainer/interfaces.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/trainer/trainer.hpp
    PRIVATE
        trainer.cpp
)

target_compile_features(tmm_trainer PUBLIC cxx_std_23)
target_include_directories(tmm_trainer PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)

target_link_libraries(tmm_trainer
    PUBLIC
        tmm_conf        # TrainingConfig, TMM_VERSION define
        tmm_datasets    # DatasetIterator, load_dataset
        tmm_plugins     # PluginManager
)

# Arrow headers needed by interfaces.hpp (Batch holds shared_ptr<arrow::RecordBatch>).
# Propagated PUBLIC so that consumers of tmm_trainer can use Arrow types without
# having to add their own include_directories.
if(arrow_SOURCE_DIR)
    target_include_directories(tmm_trainer PUBLIC
        "${arrow_SOURCE_DIR}/src"
        "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    )
endif()

# spdlog — kept PRIVATE; the trainer uses it internally for structured logging.
# Consumers that want direct spdlog access should link it themselves.
CPMAddPackage(URI "gh:gabime/spdlog@1.15.3" EXCLUDE_FROM_ALL YES)
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tmm_trainer PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)

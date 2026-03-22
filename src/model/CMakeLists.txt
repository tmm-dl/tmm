add_library(tmm_model)

target_sources(tmm_model
    PUBLIC
        FILE_SET tmm_model_headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../include
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/model/device.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/model/params.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/model/preprocessor.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/model/collator.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/model/model.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/tmm/model/pipeline.hpp
    PRIVATE
        device.cpp
        collator.cpp
        pipeline.cpp
)

target_compile_features(tmm_model PUBLIC cxx_std_23)
target_include_directories(tmm_model PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)

# tmm_trainer pulls in tmm_plugins (DLPack, WAMR, …), tmm_datasets (Arrow),
# tmm_conf, and the Arrow headers transitively.
target_link_libraries(tmm_model
    PUBLIC
        tmm_trainer   # IModel, Batch, StepOutput — used in model.hpp / pipeline.hpp
        tmm_plugins   # IModelLoader, ITransform, PluginManager, DLPack
)

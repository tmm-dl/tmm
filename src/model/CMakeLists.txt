add_library(ttm_model)

target_sources(ttm_model
    PUBLIC
        FILE_SET ttm_model_headers
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../include
        FILES
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/model/device.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/model/params.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/model/preprocessor.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/model/collator.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/model/model.hpp
            ${CMAKE_CURRENT_LIST_DIR}/../../include/ttm/model/pipeline.hpp
    PRIVATE
        device.cpp
        collator.cpp
        pipeline.cpp
)

target_compile_features(ttm_model PUBLIC cxx_std_23)
target_include_directories(ttm_model PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)

# ttm_trainer pulls in ttm_plugins (DLPack, WAMR, …), ttm_datasets (Arrow),
# ttm_conf, and the Arrow headers transitively.
target_link_libraries(ttm_model
    PUBLIC
        ttm_trainer   # IModel, Batch, StepOutput — used in model.hpp / pipeline.hpp
        ttm_plugins   # IModelLoader, ITransform, PluginManager, DLPack
)

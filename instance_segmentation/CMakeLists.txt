cmake_minimum_required(VERSION 3.13)
project(instance_mask_rcnn)

add_subdirectory(include)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

add_executable(${PROJECT_NAME} 
            main.cpp 
            # src_original/classification.cpp
            )
target_include_directories(${PROJECT_NAME} 
    PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} 
    PUBLIC ${OpenCV_INCLUDE_DIRS} 
    #PUBLIC ${QUEUE_INCLUDE_DIR} 
    PUBLIC include
    )

target_link_directories(${PROJECT_NAME} 
    PUBLIC include
)

target_link_libraries(${PROJECT_NAME} 
    PRIVATE ${ONNX_RUNTIME_LIB} 
    PUBLIC ${OpenCV_LIBRARIES} 
    #PUBLIC ${QUEUE_LINK}
    PUBLIC instance_lib 
    PUBLIC CVPLOT
    )

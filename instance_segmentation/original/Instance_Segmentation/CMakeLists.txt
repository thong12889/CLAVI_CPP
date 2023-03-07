cmake_minimum_required(VERSION 3.13)
project(instance_onnx_cpp)


find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)


add_executable(${PROJECT_NAME} instance_seg.cpp )
target_include_directories(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${QUEUE_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES} ${QUEUE_LINK})

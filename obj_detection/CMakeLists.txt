cmake_minimum_required(VERSION 3.13.4)
project(objDetection VERSION 1.0)

set(CMAKE_GXX_FLAGS "-Wall -Wextra -Wconversion  -pedantic -std=gnu+20 -std=c++20" )

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)



find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)


add_executable(${PROJECT_NAME} ObjectDetection.cpp ObjectDetection.h main.cpp GridAndStride.h Object.h)
target_include_directories(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})

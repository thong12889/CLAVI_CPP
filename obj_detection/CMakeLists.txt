cmake_minimum_required(VERSION 3.13.4)
project(objDetection VERSION 1.0)

set(CMAKE_GXX_FLAGS "-Wall -Wextra -Wconversion  -pedantic -std=gnu+20 -std=c++20" )

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(OpenCV REQUIRED )
find_package(PkgConfig REQUIRED)
pkg_search_module(GST REQUIRED gstreamer-1.0>=1.4
	gstreamer-sdp-1.0>=1.4
	gstreamer-video-1.0>-1.4
	gstreamer-app-1.0>=1.4
	)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)


add_executable(${PROJECT_NAME} 
    ObjectDetection.cpp 
    ObjectDetection.h 
    main.cpp 
    GridAndStride.h 
    Object.h 
    CVQueue.h
    CVQueue.cpp
    )
target_include_directories(${PROJECT_NAME} 
    PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} 
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    PRIVATE ${GST_INCLUDE_DIRS}
    )
target_link_libraries(${PROJECT_NAME} 
    PRIVATE ${ONNX_RUNTIME_LIB} 
    PRIVATE ${OpenCV_LIBRARIES}
    PRIVATE ${GST_LIBRARIES} gstapp-1.0
    )

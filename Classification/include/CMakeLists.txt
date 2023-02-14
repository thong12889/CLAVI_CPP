cmake_minimum_required(VERSION 3.13)
project(classify_onnx_cpp)

add_library(ObjCls ObjectClassification.cpp ObjectClassification.h )

#Find LIB via pkg-config
find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

#Link target directories to include
target_include_directories(ObjCls
PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} 
    PUBLIC ${OpenCV_INCLUDE_DIRS} 
    )
    
target_link_libraries(ObjCls PUBLIC ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} )

target_link_libraries(ObjCls
    PRIVATE ${ONNX_RUNTIME_LIB} 
    PUBLIC ${OpenCV_LIBRARIES} 
    )


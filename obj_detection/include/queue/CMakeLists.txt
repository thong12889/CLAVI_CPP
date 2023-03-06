cmake_minimum_required(VERSION 3.13.4)
project(QUEUELIB)

add_library(${PROJECT_NAME} CVQueue.cpp CVQueue.h )

find_package(OpenCV REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBRARIES})

cmake_minimum_required(VERSION 3.13.4)
project(CVPLOT)

add_library(${PROJECT_NAME} SHARED PerformancePlot.cpp PerformancePlot.h csvfile.h FPS.cpp CustomData.cpp )

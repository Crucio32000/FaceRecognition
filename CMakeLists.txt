# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.1)

# The following folder will be included

# Project's name
project(hello)
include("C:/BEWARE/DLIB/dlib/cmake")

# BOOST LIBS
set(BOOST_LIBRARYDIR $ENV{BOOST_ROOT}\\stage\\lib)
set(BOOST_INCLUDEDIR $ENV{BOOST_ROOT}\\stage\\include)
find_package(Boost REQUIRED filesystem system)


# OPENCV LIBS  
set(OpenCV_DIR C:\\BEWARE\\OpenCV\\build)
find_package(OpenCV 3 REQUIRED)


# Exec and target libraries
include_directories(${CMAKE_CURRENT_SOURCE_DIR}, ${BOOST_INCLUDEDIR})
add_executable(SVM ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries( SVM ${OpenCV_LIBS} )
target_link_libraries( SVM ${Boost_LIBRARIES})
target_link_libraries( SVM dlib )
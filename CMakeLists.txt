################################################################################
### Header
################################################################################
cmake_minimum_required(VERSION 2.8)
project(steam)


# build tool
find_package(catkin_optional REQUIRED)
co_build_type() # cache the build type

################################################################################
### Compiler Setup
################################################################################

if (CMAKE_COMPILER_IS_GNUCC)
  execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
  if (GCC_VERSION VERSION_GREATER 4.8 OR GCC_VERSION VERSION_EQUAL 4.8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    message(STATUS "C++11 capable compiler detected")
  else()
    # gcc46 does not support C++11, and the linker is too old to know about AVX instructions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -mno-avx")
    message(STATUS "C++11 support not available; defaulting to C++0x")
  endif()
else()
  message(WARNING "Cannot detect correct settings for your compiler")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

################################################################################
### Configure Dependency Headers and Libraries
################################################################################

# Add the search path for FindPackage.cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# initialize
set(EXTERNAL_LIBS "")

# find Eigen
find_package(Eigen3 3.2.2 REQUIRED)
message(STATUS "FOUND: Eigen3 version ${EIGEN3_VERSION} found in ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR})

# find OpenMP - multithreading package
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# find Lie Group Math
find_package(lgmath REQUIRED)
include_directories(${lgmath_INCLUDE_DIRS})
list(APPEND EXTERNAL_LIBS ${lgmath_LIBRARIES})

# find Boost (used for executables only)
find_package(Boost REQUIRED system)
include_directories(${Boost_INCLUDE_DIRS})
set(DEPEND_LIBS ${Boost_LIBRARIES})

################################################################################
### Enable find_package()
################################################################################

# export the package (for package config)
co_export(
  VERSION 0.2.0
  INCLUDE_DIRS include
  LIBRARIES steam
  DEPENDS lgmath Eigen3 OpenMP Boost)

################################################################################
### Specify Build Targets
################################################################################

##### Main Library 

# Find the header and source files
file(GLOB_RECURSE HEADER_FILES "include/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

# build and link the library
add_library(steam SHARED ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(steam ${EXTERNAL_LIBS})
set_target_properties(steam PROPERTIES VERSION "${PROJECT_VERSION}" SOVERSION 1)

################################################################################
### Samples and Tests
################################################################################

# add 'Catch' test directory
set(TESTS_ON TRUE CACHE BOOL "Configure unit testing (uses Catch unit tests)")
if(${TESTS_ON})
    enable_testing(true)
    add_subdirectory(tests)
endif()

# add samples directory
set(SAMPLES_ON TRUE CACHE BOOL "Build sample executables (e.g. speed test)")
if(${SAMPLES_ON})
  add_subdirectory(samples)
endif()

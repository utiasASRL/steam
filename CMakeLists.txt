################################################################################
### Header
################################################################################
cmake_minimum_required(VERSION 2.8)
project(steam)

# Allow CMake 3.0+ to run without warnings
if(NOT CMAKE_MAJOR_VERSION LESS 3)
  cmake_policy(SET CMP0048 OLD)
endif()

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

################################################################################
### Misc
################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# check that a build type was chosen, default to release
if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

################################################################################
### Library Version
################################################################################
set(MAJOR_VERSION 0)
set(MINOR_VERSION 2)
set(PATCH_VERSION 0)
set(PROJECT_VERSION ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})

################################################################################
### Configure Dependency Headers and Libraries
################################################################################

# Add the search path for FindPackage.cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# initialize
set(EXTERNAL_LIBS "")

# find catkin (optional)
string(COMPARE EQUAL "${CATKIN_DEVEL_PREFIX}" "" CATKIN_DISABLED_DEF)
set(CATKIN_DISABLED ${CATKIN_DISABLED_DEF} CACHE BOOL "Disable catkin, even if found")
if(NOT CATKIN_DISABLED)
  find_package(catkin)
  set(USE_CATKIN ${catkin_FOUND})
  if(NOT ${catkin_FOUND})
    message("Warning: catkin enabled but not found, using vanilla cmake.")
  endif()
else()
  set(USE_CATKIN FALSE)
endif()

# find Eigen
find_package(Eigen3 3.2.2 REQUIRED)
if(EIGEN3_FOUND)
  message(STATUS "FOUND: Eigen3 version ${EIGEN3_VERSION} found in ${EIGEN3_INCLUDE_DIR}")
  include_directories(${EIGEN3_INCLUDE_DIR})
else(EIGEN3_FOUND)
  message(STATUS "NOT FOUND: Eigen")
endif(EIGEN3_FOUND)

# find OpenMP - multithreading package
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "FOUND: OPENMP")
else(OPENMP_FOUND)
  message(STATUS "WARNING: OPENMP NOT FOUND - MULTICORE PERFORMANCE IS MUCH BETTER!")
endif(OPENMP_FOUND)

# find Lie Group Math
find_package(lgmath REQUIRED)
message(${lgmath_INCLUDE_DIRS})
include_directories(${lgmath_INCLUDE_DIRS})
list(APPEND EXTERNAL_LIBS ${lgmath_LIBRARIES})

# find Boost (used for executables only)
find_package(Boost REQUIRED system)
include_directories(${Boost_INCLUDE_DIRS})
set(DEPEND_LIBS ${Boost_LIBRARIES})

################################################################################
### Catkin Package (optional)
################################################################################

# export the package (for catkin Package Config)
if(USE_CATKIN)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES steam
    CATKIN_DEPENDS lgmath
    DEPENDS Eigen3 OpenMP Boost)
endif()

################################################################################
### Installation Paths
################################################################################

# Allow CMake 3.0+ to run without warnings (as of 3.0+, setting install locations per target is deprecated)
if(NOT CMAKE_MAJOR_VERSION LESS 3)
  cmake_policy(SET CMP0026 OLD)
endif()

# catkin provides install paths
if(USE_CATKIN)
  set(DEF_INSTALL_LIB_DIR ${CATKIN_PACKAGE_LIB_DESTINATION})
  set(DEF_INSTALL_BIN_DIR ${CATKIN_PACKAGE_BIN_DESTINATION})
  set(DEF_INSTALL_INCLUDE_DIR ${CATKIN_GLOBAL_INCLUDE_DESTINATION})
  set(DEF_INSTALL_CMAKE_DIR ${CATKIN_PACKAGE_SHARE_DESTINATION})
# sane default install paths for vanilla cmake
else()
  set(DEF_INSTALL_LIB_DIR lib)
  set(DEF_INSTALL_BIN_DIR bin)
  set(DEF_INSTALL_INCLUDE_DIR include)
  if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
  else()
    set(DEF_INSTALL_CMAKE_DIR lib/cmake/steam)
  endif()
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR ${DEF_INSTALL_LIB_DIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR ${DEF_INSTALL_BIN_DIR} CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR ${DEF_INSTALL_INCLUDE_DIR} CACHE PATH "Installation directory for header files")
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

################################################################################
### Configure Source Files and Include Directories
################################################################################

# Tell CMake where the headers are
set(HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories(${HEADER_DIR})

# Find the header files
file(GLOB_RECURSE HEADER_FILES "${HEADER_DIR}/*.hpp")

# Find the source files
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

################################################################################
### Specify Build Targets
################################################################################

##### Main Library 
# Tell cmake about the library
add_library(steam SHARED ${SOURCE_FILES} ${HEADER_FILES})
# Link against the required libraries
target_link_libraries(steam ${EXTERNAL_LIBS})

## install targets
set_target_properties(steam PROPERTIES VERSION "${PROJECT_VERSION}" SOVERSION 1)
install(TARGETS steam
  LIBRARY DESTINATION ${INSTALL_LIB_DIR}
)
install(DIRECTORY ${HEADER_DIR}/
  DESTINATION ${INSTALL_INCLUDE_DIR}
)

##### Sample Executables

# samples
set(SAMPLES_ON TRUE CACHE BOOL "Build sample executables (e.g. speed test)")
if(${SAMPLES_ON})
  add_subdirectory(samples)
endif()

################################################################################
### Unit Tests
################################################################################

# Add 'Catch' test directory
# Turn tests on and off using the CMake GUI
set(TESTS_ON TRUE CACHE BOOL "Configure unit testing (uses Catch unit tests)")
if(${TESTS_ON})
    enable_testing(true)
    add_subdirectory(tests)
endif()

################################################################################
### Enable find_package()
################################################################################

if(NOT USE_CATKIN) # Catkin handles all of this in catkin_package()

  ##### Create the PackageConfig.cmake and PackageConfigVersion files

  ### local build-tree

  # Export the package for use from the build-tree
  # (registers the build-tree with a global CMake-registry)
  export(PACKAGE steam)

  file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")

  # Create variable for the local build tree
  get_property(CONF_INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

  # Create variable with the library location
  get_target_property(STEAM_LIB steam LOCATION)

  # Create the <name>Config.cmake for local build tree
  configure_file(steamConfig.cmake.in
    "${PROJECT_BINARY_DIR}/steamConfig.cmake" @ONLY)

  ### install build-tree

  # Change the include location for the case of an install location
  set(CONF_INCLUDE_DIRS ${INSTALL_INCLUDE_DIR} ${CONF_INCLUDE_DIRS} )

  # gather all the includes but remove ones in the source tree
  list(REMOVE_ITEM CONF_INCLUDE_DIRS ${HEADER_DIR})

  # Change the library location for an install location
  get_filename_component(STEAM_LIB_NAME ${STEAM_LIB} NAME)
  set(STEAM_LIB ${INSTALL_LIB_DIR}/${STEAM_LIB_NAME})

  # Create the <name>Config.cmake for the installation tree
  configure_file(steamConfig.cmake.in
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/steamConfig.cmake" @ONLY)

  # The same versioning file can be used for both local and instal trees
  configure_file(steamConfigVersion.cmake.in
    "${PROJECT_BINARY_DIR}/steamConfigVersion.cmake" @ONLY)

  # Install the SteamConfig.cmake and SteamConfigVersion.cmake
  install(FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/steamConfig.cmake"
    "${PROJECT_BINARY_DIR}/steamConfigVersion.cmake"
    DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

  ################################################################################
  ### Uninstall Target
  ################################################################################

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

endif()

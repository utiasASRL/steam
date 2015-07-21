################################################################################
### Header
################################################################################
cmake_minimum_required(VERSION 2.8)
project(steam)

# define number of threads to be used by OpenMP
add_definitions(-DNUMBER_OF_OPENMP_THREADS=8)

################################################################################
### Misc
################################################################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mssse3 -mfpmath=sse -msse -msse2 -mmmx -msse3")

# if it does not exist, add a cmake directory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()

# check that a build type was chosen, default to release
if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")

################################################################################
### Library Version
################################################################################
set(MAJOR_VERSION 0)
set(MINOR_VERSION 1)
set(PATCH_VERSION 0)
set(PROJECT_VERSION ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})

################################################################################
### Installation Paths
################################################################################

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake/steam)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

################################################################################
### Configure Dependency Headers and Libraries
################################################################################

# initialize
set(EXTERNAL_LIBS "")

# find Eigen
find_package(Eigen3 3.2.2 REQUIRED)
if(EIGEN3_FOUND)
  message(STATUS "FOUND: Eigen3 version ${EIGEN3_VERSION} found in ${EIGEN3_INCLUDE_DIR}")
  include_directories(${EIGEN3_INCLUDE_DIR})
else(EIGEN3_FOUND)
  message(STATUS "NOT FOUND: Eigen")
endif(EIGEN3_FOUND)

# find Glog
find_package(Glog REQUIRED)
if(GLOG_FOUND)
  message(STATUS "FOUND: Glog at ${GLOG_LIBRARIES}")
  include_directories(${GLOG_INCLUDE_DIRS})
  set(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${GLOG_LIBRARIES})
else(GLOG_FOUND)
  message(STATUS "NOT FOUND: Glog")
endif(GLOG_FOUND)

# find OpenMP - multithreading package
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS "FOUND: OPENMP")
else(OPENMP_FOUND)
  message(STATUS "NOT FOUND: OPENMP")
endif(OPENMP_FOUND)

# find Lie Group Math
find_package(LGMath REQUIRED)
include_directories(${LGMath_INCLUDE_DIRS})
set(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${LGMath_LIBRARIES})

################################################################################
### Configure Source Files and Include Directories
################################################################################

# Find the header files
file(GLOB_RECURSE HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/steam/*.hpp")

# Find the sources files
file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Tell CMake where the headers are
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

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
install(FILES
  ${HEADER_FILES}
  DESTINATION ${INSTALL_INCLUDE_DIR}/steam
)

##### Sample Executables

# samples
add_subdirectory(samples)

################################################################################
### Unit Tests
################################################################################
# Add 'Catch' test directory
# Turn tests on and off using the CMake GUI
set(TESTS_ON true CACHE BOOL "Configure unit testing (uses Catch unit tests)")
if(TESTS_ON)
    enable_testing(true) 
    add_subdirectory(tests)
endif(TESTS_ON)


################################################################################
### Enable find_package()
################################################################################

##### Create the SteamConfig.cmake and SteamConfigVersion files

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
configure_file(SteamConfig.cmake.in
  "${PROJECT_BINARY_DIR}/SteamConfig.cmake" @ONLY)

### install build-tree

# Change the include location for the case of an install location
set(CONF_INCLUDE_DIRS ${INSTALL_INCLUDE_DIR} ${CONF_INCLUDE_DIRS} )

# gather all the includes but remove ones in the source tree
list(REMOVE_ITEM CONF_INCLUDE_DIRS ${CMAKE_SOURCE_DIR})

# Change the library location for an install location
get_filename_component(STEAM_LIB_NAME ${STEAM_LIB} NAME)
set(STEAM_LIB ${INSTALL_LIB_DIR}/${STEAM_LIB_NAME})

# Create the <name>Config.cmake for the installation tree
configure_file(SteamConfig.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SteamConfig.cmake" @ONLY)

# The same versioning file can be used for both local and instal trees
configure_file(SteamConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}/SteamConfigVersion.cmake" @ONLY)

# Install the SteamConfig.cmake and SteamConfigVersion.cmake
install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SteamConfig.cmake"
  "${PROJECT_BINARY_DIR}/SteamConfigVersion.cmake"
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



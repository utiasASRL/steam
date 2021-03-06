cmake_minimum_required(VERSION 3.16)
project(steam)

option(USE_AMENT "Use ament_cmake to build lgmath for ROS2." OFF)

# Compiler setup (assumed to be GNU)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-march=native -O3 -Wall -pedantic -Wno-unused-function)

# define number of threads to be used by OpenMP in steam
add_definitions(-DSTEAM_DEFAULT_NUM_OPENMP_THREADS=4)
# Use a object pool to improve performance but no longer thread safe as the object pool
# is a singleton and is based on openmp. There is about ~10 gain in speed.
# add_definitions(-DSTEAM_USE_OBJECT_POOL)

## cmake flow (default)
if (NOT USE_AMENT)

set(PROJECT_VERSION 1.1.0)

# Find dependencies
find_package(Boost 1.71.0 REQUIRED system)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(Eigen3 3.3.7 REQUIRED)
find_package(lgmath 1.1.0 REQUIRED)

# Build library
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${Boost_LIBRARIES}
    lgmath
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
  EXPORT ${PROJECT_NAME}Targets
)

# Export
set(PROJECT_LIBRARY ${PROJECT_NAME})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
# export to build directory so no need to install in order to use find_package
export(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake
)

# install export cmake files
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION lib/cmake/${PROJECT_NAME}
)

## ROS2 ament_cmake flow
else()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED system)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(lgmath REQUIRED)

# Libraries
file(GLOB_RECURSE SOURCE src/*.cpp)
add_library(${PROJECT_NAME} ${SOURCE})
ament_target_dependencies(${PROJECT_NAME} Boost Eigen3 OpenMP lgmath)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)


ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
# May need the following
ament_export_dependencies(
  Boost
  Eigen3
  OpenMP
  lgmath
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
    # Libraries
    ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

endif()

# Executables
add_executable(SimpleBAandTrajPrior samples/SimpleBAandTrajPrior.cpp)
target_link_libraries(SimpleBAandTrajPrior ${PROJECT_NAME})

add_executable(SimpleBAandCATrajPrior samples/SimpleBAandCATrajPrior.cpp)
target_link_libraries(SimpleBAandCATrajPrior ${PROJECT_NAME})

add_executable(SimpleBundleAdjustment samples/SimpleBundleAdjustment.cpp)
target_link_libraries(SimpleBundleAdjustment ${PROJECT_NAME})

add_executable(SimpleBundleAdjustmentFullRel samples/SimpleBundleAdjustmentFullRel.cpp)
target_link_libraries(SimpleBundleAdjustmentFullRel ${PROJECT_NAME})

add_executable(SimpleBundleAdjustmentRelLand samples/SimpleBundleAdjustmentRelLand.cpp)
target_link_libraries(SimpleBundleAdjustmentRelLand ${PROJECT_NAME})

add_executable(SimpleBundleAdjustmentRelLandX samples/SimpleBundleAdjustmentRelLandX.cpp)
target_link_libraries(SimpleBundleAdjustmentRelLandX ${PROJECT_NAME})

add_executable(SimplePoseGraphRelax samples/SimplePoseGraphRelax.cpp)
target_link_libraries(SimplePoseGraphRelax ${PROJECT_NAME})

add_executable(SimpleTrajectoryPrior samples/SimpleTrajectoryPrior.cpp)
target_link_libraries(SimpleTrajectoryPrior ${PROJECT_NAME})

add_executable(SpherePoseGraphRelax samples/SpherePoseGraphRelax.cpp)
target_link_libraries(SpherePoseGraphRelax ${PROJECT_NAME})

add_executable(TrustRegionExample samples/TrustRegionExample.cpp)
target_link_libraries(TrustRegionExample ${PROJECT_NAME})

add_executable(MotionDistortedP2PandCATrajPrior samples/MotionDistortedP2PandCATrajPrior.cpp)
target_link_libraries(MotionDistortedP2PandCATrajPrior ${PROJECT_NAME})

add_executable(SimpleP2PandCATrajPrior samples/SimpleP2PandCATrajPrior.cpp)
target_link_libraries(SimpleP2PandCATrajPrior ${PROJECT_NAME})

if (USE_AMENT)

install(
  TARGETS
    # Executables
    SimpleBAandTrajPrior
    SimpleBAandCATrajPrior
    SimpleBundleAdjustment
    SimpleBundleAdjustmentFullRel
    SimpleBundleAdjustmentRelLand
    SimpleBundleAdjustmentRelLandX
    SimplePoseGraphRelax
    SimpleTrajectoryPrior
    SpherePoseGraphRelax
    TrustRegionExample
    MotionDistortedP2PandCATrajPrior
    SimpleP2PandCATrajPrior
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # Unit-tests
  ament_add_gtest(time_test tests/time_test.cpp)
  target_link_libraries(time_test ${PROJECT_NAME})
  ament_add_gtest(pattern_test tests/pattern_test.cpp)
  target_link_libraries(pattern_test ${PROJECT_NAME})
  ament_add_gtest(evaluator_test tests/evaluator_test.cpp)
  target_link_libraries(evaluator_test ${PROJECT_NAME})

  # Linting
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies() # Lint based on linter test_depend in package.xml
endif()

ament_package()
endif()

cmake_minimum_required(VERSION 3.16)
project(steam)

option(USE_AMENT "Use ament_cmake to build lgmath for ROS2." ON)
option(BUILD_TESTING "build tests for steam" OFF)

# Compiler setup (assumed to be GNU)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

add_compile_options(-march=native -O3 -Wall -pedantic -Wno-unused-function)

# define number of threads to be used by OpenMP in steam
add_definitions(-DSTEAM_DEFAULT_NUM_OPENMP_THREADS=4)

## cmake flow (default)
if (NOT USE_AMENT)

set(PROJECT_VERSION 1.1.0)

# Find dependencies
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(Eigen3 3.3.7 REQUIRED)
find_package(lgmath 1.1.0 REQUIRED)

# Build library
file(GLOB_RECURSE SOURCE
  src/data/*.cpp
  src/blockmat/*.cpp
  src/evaluable/*.cpp
  src/problem/*.cpp
  src/solver/*.cpp
  src/trajectory/*.cpp
)
add_library(${PROJECT_NAME} SHARED ${SOURCE})
target_link_libraries(${PROJECT_NAME}
  PUBLIC lgmath
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

if (BUILD_TESTING)
  enable_testing()
  add_executable(pattern_test tests/pattern_test.cpp)
  add_executable(time_test tests/time_test.cpp)
  add_executable(jacobian_test tests/jacobian_test.cpp)
  add_executable(br2d_test tests/br2d_error_eval_test.cpp)
  target_link_libraries(pattern_test ${PROJECT_NAME} GTest::gtest_main)
  target_link_libraries(time_test ${PROJECT_NAME} GTest::gtest_main)
  target_link_libraries(jacobian_test ${PROJECT_NAME} GTest::gtest_main)
  target_link_libraries(br2d_test ${PROJECT_NAME} GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(pattern_test)
  gtest_discover_tests(time_test)
  gtest_discover_tests(jacobian_test)
  gtest_discover_tests(br2d_test)
endif()

## ROS2 ament_cmake flow
else()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(lgmath REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# Libraries
file(GLOB_RECURSE SOURCE
  src/data/*.cpp
  src/blockmat/*.cpp
  src/evaluable/*.cpp
  src/problem/*.cpp
  src/solver/*.cpp
  src/trajectory/*.cpp
)
add_library(${PROJECT_NAME} ${SOURCE})
ament_target_dependencies(${PROJECT_NAME} Boost OpenMP Eigen3 lgmath)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(eigen3_cmake_module)
ament_export_dependencies(Boost OpenMP Eigen3 lgmath)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

endif()

# Executables
link_libraries(${PROJECT_NAME})
add_executable(BSplineTrajectoryExample samples/BSplineTrajectoryExample.cpp)
add_executable(LandmarkSLAMExample samples/LandmarkSLAMExample.cpp)
add_executable(RadialVelMeasWithConstVelTraj samples/RadialVelMeasWithConstVelTraj.cpp)
add_executable(RadialVelocityMeasurement samples/RadialVelocityMeasurement.cpp)
add_executable(SimplePoseGraphRelax samples/SimplePoseGraphRelax.cpp)
add_executable(SpherePoseGraphRelax samples/SpherePoseGraphRelax.cpp)
add_executable(SimplePointCloudAlignment samples/SimplePointCloudAlignment.cpp)
add_executable(TrustRegionExample samples/TrustRegionExample.cpp)
add_executable(SimpleConstVelSE2TrajPrior samples/SimpleConstVelSE2TrajPrior.cpp)
add_executable(SimpleConstVelTrajPrior samples/SimpleConstVelTrajPrior.cpp)
add_executable(SimpleBundleAdjustment samples/SimpleBundleAdjustment.cpp)
add_executable(SimpleBundleAdjustmentRelLand samples/SimpleBundleAdjustmentRelLand.cpp)
add_executable(SimpleBundleAdjustmentFullRel samples/SimpleBundleAdjustmentFullRel.cpp)
add_executable(SimpleBAandConstVelTrajPrior samples/SimpleBAandConstVelTrajPrior.cpp)
add_executable(SlidingWindowFilterExample samples/SlidingWindowFilterExample.cpp)
add_executable(MotionPriors samples/MotionPriors.cpp)
add_executable(LostInTheWoods samples/LostInTheWoodsSteam.cpp)
target_link_libraries(LostInTheWoods yaml-cpp)
# if(yaml-cpp_FOUND)
#   add_executable(LostInTheWoods samples/LostInTheWoodsSteam.cpp)
#   target_link_libraries(LostInTheWoods yaml-cpp::yaml-cpp)
# endif()

if (USE_AMENT)

install(
  TARGETS
    # Executables
    BSplineTrajectoryExample
    LandmarkSLAMExample
    RadialVelMeasWithConstVelTraj
    RadialVelocityMeasurement
    SimplePoseGraphRelax
    SpherePoseGraphRelax
    SimplePointCloudAlignment
    TrustRegionExample
    SimpleConstVelSE2TrajPrior
    SimpleConstVelTrajPrior
    SimpleBundleAdjustment
    SimpleBundleAdjustmentRelLand
    SimpleBundleAdjustmentFullRel
    SimpleBAandConstVelTrajPrior
    SlidingWindowFilterExample
    MotionPriors
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
  ament_add_gtest(jacobian_test tests/jacobian_test.cpp)
  target_link_libraries(jacobian_test ${PROJECT_NAME})
  ament_add_gtest(br2d_test tests/br2d_error_eval_test.cpp)
  target_link_libraries(br2d_test ${PROJECT_NAME})

  # Linting
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies() # Lint based on linter test_depend in package.xml
endif()

ament_package()
endif()

cmake_minimum_required(VERSION 3.16)
project(steam-samples)

# Compiler setup (assumed to be GNU)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-march=native -O3 -Wall -pedantic -Wno-unused-function)

# Boost and OpenMP
find_package(Boost 1.71.0 REQUIRED system)
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Find and link to lgmath and steam
find_package(steam REQUIRED)
link_libraries(steam)  # link to all targets below, no need to specify include directories.

add_executable(SimpleBAandTrajPrior SimpleBAandTrajPrior.cpp)
add_executable(SimpleBAandCATrajPrior SimpleBAandCATrajPrior.cpp)
add_executable(SimpleBundleAdjustment SimpleBundleAdjustment.cpp)
add_executable(SimpleBundleAdjustmentFullRel SimpleBundleAdjustmentFullRel.cpp)
add_executable(SimpleBundleAdjustmentRelLand SimpleBundleAdjustmentRelLand.cpp)
add_executable(SimpleBundleAdjustmentRelLandX SimpleBundleAdjustmentRelLandX.cpp)
add_executable(SimplePoseGraphRelax SimplePoseGraphRelax.cpp)
add_executable(SimpleTrajectoryPrior SimpleTrajectoryPrior.cpp)
add_executable(SpherePoseGraphRelax SpherePoseGraphRelax.cpp)
add_executable(TrustRegionExample TrustRegionExample.cpp)
add_executable(MotionDistortedP2PandCATrajPrior MotionDistortedP2PandCATrajPrior.cpp)
add_executable(SimpleP2PandCATrajPrior SimpleP2PandCATrajPrior.cpp)
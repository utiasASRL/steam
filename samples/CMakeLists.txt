################################################################################
### Sample Programs
################################################################################

# define number of threads to be used by OpenMP in steam
add_definitions(-DSTEAM_DEFAULT_NUM_OPENMP_THREADS=4)

# convergence sample
add_executable(TrustRegionExample ${CMAKE_CURRENT_SOURCE_DIR}/TrustRegionExample.cpp)
target_link_libraries(TrustRegionExample steam boost_system ${DEPEND_LIBS})

# simple pose graph relaxation example
add_executable(SimplePoseGraphRelax ${CMAKE_CURRENT_SOURCE_DIR}/SimplePoseGraphRelax.cpp)
target_link_libraries(SimplePoseGraphRelax steam boost_system ${DEPEND_LIBS})

# spherical pose graph example
add_executable(SpherePoseGraphRelax ${CMAKE_CURRENT_SOURCE_DIR}/SpherePoseGraphRelax.cpp)
target_link_libraries(SpherePoseGraphRelax steam boost_system ${DEPEND_LIBS})

# bundle adjustment example with hardcoded GP factors
add_executable(SimpleBundleAdjustment ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustment.cpp)
target_link_libraries(SimpleBundleAdjustment steam boost_system ${DEPEND_LIBS})

# bundle adjustment example with relative landmarks
add_executable(SimpleBundleAdjustmentRelLand ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustmentRelLand.cpp)
target_link_libraries(SimpleBundleAdjustmentRelLand steam boost_system ${DEPEND_LIBS})

# bundle adjustment example with fully relative landmarks and poses
add_executable(SimpleBundleAdjustmentFullRel ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustmentFullRel.cpp)
target_link_libraries(SimpleBundleAdjustmentFullRel steam boost_system ${DEPEND_LIBS})

# bundle adjustment example with hardcoded GP factors
add_executable(SimpleBundleAdjustmentRelLandX ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustmentRelLandX.cpp)
target_link_libraries(SimpleBundleAdjustmentRelLandX steam boost_system ${DEPEND_LIBS})

# simple example of trajectory interface for prior
add_executable(SimpleTrajectoryPrior ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTrajectoryPrior.cpp)
target_link_libraries(SimpleTrajectoryPrior steam boost_system ${DEPEND_LIBS})

# simple example of trajectory interface for prior, combined with bundle adjustment
add_executable(SimpleBAandTrajPrior ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBAandTrajPrior.cpp)
target_link_libraries(SimpleBAandTrajPrior steam boost_system ${DEPEND_LIBS})

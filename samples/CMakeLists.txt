################################################################################
### Sample Programs
################################################################################

# define number of threads to be used by OpenMP in steam
add_definitions(-DSTEAM_DEFAULT_NUM_OPENMP_THREADS=4)

# convergence sample
add_executable(TrustRegionExample ${CMAKE_CURRENT_SOURCE_DIR}/TrustRegionExample.cpp)
target_link_libraries(TrustRegionExample steam ${DEPEND_LIBS})

# simple pose graph relaxation example
add_executable(SimplePoseGraphRelax ${CMAKE_CURRENT_SOURCE_DIR}/SimplePoseGraphRelax.cpp)
target_link_libraries(SimplePoseGraphRelax steam ${DEPEND_LIBS})

# spherical pose graph example
add_executable(SpherePoseGraphRelax ${CMAKE_CURRENT_SOURCE_DIR}/SpherePoseGraphRelax.cpp)
target_link_libraries(SpherePoseGraphRelax steam ${DEPEND_LIBS})

# bundle adjustment example with hardcoded GP factors
add_executable(SimpleBundleAdjustment ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustment.cpp)
target_link_libraries(SimpleBundleAdjustment steam ${DEPEND_LIBS})

# bundle adjustment example with relative landmarks
add_executable(SimpleBundleAdjustmentRelLand ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustmentRelLand.cpp)
target_link_libraries(SimpleBundleAdjustmentRelLand steam ${DEPEND_LIBS})

# bundle adjustment example with fully relative landmarks and poses
add_executable(SimpleBundleAdjustmentFullRel ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustmentFullRel.cpp)
target_link_libraries(SimpleBundleAdjustmentFullRel steam ${DEPEND_LIBS})

# bundle adjustment example with hardcoded GP factors
add_executable(SimpleBundleAdjustmentRelLandX ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustmentRelLandX.cpp)
target_link_libraries(SimpleBundleAdjustmentRelLandX steam ${DEPEND_LIBS})

# simple example of trajectory interface for prior
add_executable(SimpleTrajectoryPrior ${CMAKE_CURRENT_SOURCE_DIR}/SimpleTrajectoryPrior.cpp)
target_link_libraries(SimpleTrajectoryPrior steam ${DEPEND_LIBS})

# simple example of trajectory interface for prior, combined with bundle adjustment
add_executable(SimpleBAandTrajPrior ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBAandTrajPrior.cpp)
target_link_libraries(SimpleBAandTrajPrior steam ${DEPEND_LIBS})

# simple example of constant acceleration trajectory interface for prior, combined with bundle adjustment
add_executable(SimpleBAandCATrajPrior ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBAandCATrajPrior.cpp)
target_link_libraries(SimpleBAandCATrajPrior steam ${DEPEND_LIBS})

# simple example of constant acceleration trajectory interface for prior, combined with p2p error terms
add_executable(SimpleP2PandCATrajPrior ${CMAKE_CURRENT_SOURCE_DIR}/SimpleP2PandCATrajPrior.cpp)
target_link_libraries(SimpleP2PandCATrajPrior steam ${DEPEND_LIBS})

# simple example of constant acceleration trajectory interface for prior, combined with p2p error terms
add_executable(MotionDistortedP2PandCATrajPrior ${CMAKE_CURRENT_SOURCE_DIR}/MotionDistortedP2PandCATrajPrior.cpp)
target_link_libraries(MotionDistortedP2PandCATrajPrior steam ${DEPEND_LIBS})

# simple example of singer acceleration model trajectory interface for prior, combined with bundle adjustment
add_executable(SimpleBAandSingerTrajPrior ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBAandSingerTrajPrior.cpp)
target_link_libraries(SimpleBAandSingerTrajPrior steam ${DEPEND_LIBS})

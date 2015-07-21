################################################################################
### Sample Programs
################################################################################

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

# bundle adjustment example with hardcoded GP factors
add_executable(SimpleBundleAdjustmentRelLand ${CMAKE_CURRENT_SOURCE_DIR}/SimpleBundleAdjustmentRelLand.cpp)
target_link_libraries(SimpleBundleAdjustmentRelLand steam ${DEPEND_LIBS})

# bundle adjustment example with hardcoded GP factors
add_executable(NaiveGPandBA ${CMAKE_CURRENT_SOURCE_DIR}/NaiveGPandBA.cpp)
target_link_libraries(NaiveGPandBA steam ${DEPEND_LIBS})

################################################################################
### Header
################################################################################
cmake_minimum_required(VERSION 2.8)

# Allow CMake 3.0+ to run without warnings
if(NOT CMAKE_MAJOR_VERSION LESS 3)
  cmake_policy(SET CMP0048 NEW)
endif()

project(catch_tests)

################################################################################
### Misc
################################################################################

# Find Catch (the unit-test header-only library)
co_find(catch)

# Disable compiler warnings that Catch throws
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-variadic-macros")
elseif(MSVC)
endif()


################################################################################
### Specify Build Targets
################################################################################

# Build test executable
add_executable(steam_unit_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sample_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/time_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pattern_test.cpp
)
target_link_libraries(steam_unit_tests steam ${DEPEND_LIBS})

# Add cmake unit tests
add_test(NAME steam_unit_tests COMMAND steam_unit_tests)

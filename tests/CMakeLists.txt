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
# Expose required variable (CATCH_INCLUDE_DIR) to parent scope
set(CATCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty"
        CACHE INTERNAL "Path to include folder for Catch")

# Disable compiler warnings that Catch throws
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-variadic-macros")
elseif(MSVC)
endif()


################################################################################
### Specify Build Targets
################################################################################
include_directories(${CATCH_INCLUDE_DIR})

# Build test executable
add_executable(steam_unit_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sample_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/time_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pattern_test.cpp
)
target_link_libraries(steam_unit_tests steam boost_system ${DEPEND_LIBS})

# Add cmake unit tests
add_test(NAME steam_unit_tests COMMAND steam_unit_tests)

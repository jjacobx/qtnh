cmake_minimum_required(VERSION 3.13.0)

project(QTNH 
  VERSION 0.0.1
  DESCRIPTION "A quantum computer simulator via generalised tensor network contraction"
  LANGUAGES C CXX
)

set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(SKIP_ARCHER2_LIBS "Skips libraries that don't require linking on ARCHER2" OFF)


## Dependencies

# Math
find_library(MATH_LIBRARY m REQUIRED)

# MPI
find_package(MPI COMPONENTS C CXX REQUIRED)

# ScaLAPACK
if (NOT SKIP_ARCHER2_LIBS)
  find_package(ScaLAPACK REQUIRED)
endif()

## QTNH
add_subdirectory(src)

## Tests
option(BUILD_TESTS "Build tests" ON)

if (BUILD_TESTS)
  find_package(Catch2 3.1.0 REQUIRED)
  enable_testing()
  include(CTest)
  include(Catch)
  add_subdirectory(tests)
else()
  message(STATUS "Skipping tests")
endif()

## Examples
add_subdirectory(examples)

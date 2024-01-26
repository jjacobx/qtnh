cmake_minimum_required(VERSION 3.10)

project(GTNH 
  VERSION 0.0.1
  DESCRIPTION "A quantum computer simulator via generalised tensor network contraction"
  LANGUAGES CXX
)

set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build shared library" ON)


## Dependencies

# Math
find_library(MATH_LIBRARY m REQUIRED)

# OpenMP
find_package(OpenMP)

# MPI
find_package(MPI COMPONENTS CXX REQUIRED)

## GTNH
add_subdirectory(src)

## Tests
find_package(Catch2 3.1.0 REQUIRED)
enable_testing()
include(CTest)
include(Catch)

add_subdirectory(tests)

## Examples
add_subdirectory(examples)

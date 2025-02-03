add_executable(distribution-test distribution.cpp)
add_dependencies(distribution-test qtnh)
target_link_libraries(distribution-test qtnh)
target_compile_options(distribution-test
  PRIVATE
  -Wall
)

add_executable(scalapack-test scalapack.cpp)
add_dependencies(scalapack-test qtnh)
target_link_libraries(scalapack-test qtnh)
target_compile_options(scalapack-test
  PRIVATE
  -Wall
)

add_executable(mpi-test mpi.cpp)
add_dependencies(mpi-test qtnh)
target_link_libraries(mpi-test qtnh)
target_compile_options(mpi-test
  PRIVATE
  -Wall
)

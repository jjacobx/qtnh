add_executable(distribution-test distribution.cpp)
add_dependencies(distribution-test qtnh)
target_link_libraries(distribution-test qtnh)
target_compile_options(distribution-test
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

add_executable(svd-test svd.cpp)
add_dependencies(svd-test qtnh)
target_link_libraries(svd-test qtnh)
target_compile_options(svd-test
  PRIVATE
  -Wall
)
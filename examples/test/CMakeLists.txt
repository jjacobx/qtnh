add_executable(distribution-test distribution.cpp)
add_dependencies(distribution-test qtnh)
target_link_libraries(distribution-test qtnh)
target_compile_options(distribution-test
  PRIVATE
  -Wall
)

add_executable(gemm-test gemm.cpp)
add_dependencies(gemm-test qtnh)
target_link_libraries(gemm-test qtnh)
target_compile_options(gemm-test
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

add_executable(bcmps-test bcmps.cpp)
add_dependencies(bcmps-test qtnh)
target_link_libraries(bcmps-test qtnh)
target_compile_options(bcmps-test
  PRIVATE
  -Wall
)

add_executable(mps-test mps.cpp)
add_dependencies(mps-test qtnh)
target_link_libraries(mps-test qtnh)
target_compile_options(mps-test
  PRIVATE
  -Wall
)

add_executable(serial-test serial.cpp)
add_dependencies(serial-test qtnh)
target_link_libraries(serial-test qtnh)
target_compile_options(serial-test
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

add_executable(simple-tensor
  simple-tensor.cpp
)

add_dependencies(simple-tensor
  qtnh
)

target_link_libraries(simple-tensor
  qtnh
)

target_compile_options(simple-tensor
  PRIVATE
  -Wall
)

add_executable(new-tensor
  new-tensor.cpp
)

add_dependencies(new-tensor
  qtnh
)

target_link_libraries(new-tensor
  qtnh
)

target_compile_options(new-tensor
  PRIVATE
  -Wall
)

add_executable(mpi-tests
  mpi-tests.cpp
)

add_dependencies(mpi-tests
  qtnh
)

target_link_libraries(mpi-tests
  qtnh
)

target_compile_options(mpi-tests
  PRIVATE
  -Wall
)

add_executable(qft
  qft.cpp
)

add_dependencies(qft
  qtnh
)

target_link_libraries(qft
  qtnh
)

target_compile_options(qft
  PRIVATE
  -Wall
)

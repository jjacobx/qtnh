add_executable(c2-indexing
  c2-indexing.cpp
)

target_compile_features(c2-indexing
  PUBLIC
  cxx_std_17
)

target_link_libraries(c2-indexing
  Catch2::Catch2WithMain
  qtnh
)

catch_discover_tests(c2-indexing)

add_executable(c2-tensor-tests
  c2-tensor-tests.cpp
)

target_compile_features(c2-tensor-tests
  PUBLIC
  cxx_std_17
)

target_link_libraries(c2-tensor-tests
  Catch2::Catch2WithMain
  qtnh
)

catch_discover_tests(c2-tensor-tests)


add_executable(c2-tensor-validation
  c2-tensor-validation.cpp
)

target_compile_features(c2-tensor-validation
  PUBLIC
  cxx_std_17
)

target_link_libraries(c2-tensor-validation
  Catch2::Catch2WithMain
  qtnh
)

catch_discover_tests(c2-tensor-validation)


add_executable(c2-mpi
  c2-mpi.cpp
)

target_compile_features(c2-mpi
  PUBLIC
  cxx_std_17
)

target_link_libraries(c2-mpi
  Catch2::Catch2
  qtnh
)

add_executable(c2-tensor
  c2-tensor.cpp
)

target_compile_features(c2-tensor
  PUBLIC
  cxx_std_17
)

target_link_libraries(c2-tensor
  Catch2::Catch2WithMain
  qtnh
)

catch_discover_tests(c2-tensor)

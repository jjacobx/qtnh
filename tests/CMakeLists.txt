add_executable(c2-tensor-tests
c2-tensor-tests.cpp
)

target_compile_features(c2-tensor-tests
  PUBLIC
  cxx_std_17
)

target_link_libraries(c2-tensor-tests
  Catch2::Catch2WithMain
  gtnh
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
  gtnh
)

catch_discover_tests(c2-tensor-validation)

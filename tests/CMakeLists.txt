add_executable(c2TensorTests
c2TensorTests.cpp
)

target_compile_features(c2TensorTests
  PUBLIC
  cxx_std_17
)

target_link_libraries(c2TensorTests
  Catch2::Catch2WithMain
  gtnh
)

catch_discover_tests(c2TensorTests)

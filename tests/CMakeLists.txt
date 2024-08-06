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

# add_executable(c2-mpi
#   c2-mpi.cpp
# )

# target_compile_features(c2-mpi
#   PUBLIC
#   cxx_std_17
# )

# target_link_libraries(c2-mpi
#   Catch2::Catch2WithMain
#   qtnh
# )

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

add_executable(c2-network
  c2-network.cpp
)

target_compile_features(c2-network
  PUBLIC
  cxx_std_17
)

target_link_libraries(c2-network
  Catch2::Catch2WithMain
  qtnh
)

catch_discover_tests(c2-network)

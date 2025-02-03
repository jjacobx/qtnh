add_subdirectory(bench)
add_subdirectory(test)

add_executable(simple-tensor simple-tensor.cpp)
add_dependencies(simple-tensor qtnh)
target_link_libraries(simple-tensor qtnh)
target_compile_options(simple-tensor
  PRIVATE
  -Wall
)

add_executable(new-tensor new-tensor.cpp)
add_dependencies(new-tensor qtnh)
target_link_libraries(new-tensor qtnh)
target_compile_options(new-tensor
  PRIVATE
  -Wall
)

add_executable(qft qft.cpp)
add_dependencies(qft qtnh)
target_link_libraries(qft qtnh)
target_compile_options(qft
  PRIVATE
  -Wall
)

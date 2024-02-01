add_executable(simple-tensor
  simple-tensor.cpp
)

add_dependencies(simple-tensor
  gtnh
)

target_link_libraries(simple-tensor
  gtnh
)

target_compile_options(simple-tensor
  PRIVATE
  -Wall
)

add_executable(new-tensor
  new-tensor.cpp
)

add_dependencies(new-tensor
  gtnh
)

target_link_libraries(new-tensor
  gtnh
)

target_compile_options(new-tensor
  PRIVATE
  -Wall
)

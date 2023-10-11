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

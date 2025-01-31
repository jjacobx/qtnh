add_executable(permute permute.cpp)
add_dependencies(permute qtnh)
target_link_libraries(permute qtnh)

add_executable(contract contract.cpp)
add_dependencies(contract qtnh)
target_link_libraries(contract qtnh)
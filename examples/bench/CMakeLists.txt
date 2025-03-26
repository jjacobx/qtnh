add_executable(contract contract.cpp)
add_dependencies(contract qtnh)
target_link_libraries(contract qtnh)

add_executable(mps-qft mps-qft.cpp)
add_dependencies(mps-qft qtnh)
target_link_libraries(mps-qft qtnh)

add_executable(permute permute.cpp)
add_dependencies(permute qtnh)
target_link_libraries(permute qtnh)
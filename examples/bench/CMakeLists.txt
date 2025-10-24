add_executable(contract contract.cpp)
add_dependencies(contract qtnh)
target_link_libraries(contract qtnh)

add_executable(decompose decompose.cpp)
add_dependencies(decompose qtnh)
target_link_libraries(decompose qtnh)

add_executable(mps-qft mps-qft.cpp)
add_dependencies(mps-qft qtnh)
target_link_libraries(mps-qft qtnh)

add_executable(mps-rcs mps-rcs.cpp)
add_dependencies(mps-rcs qtnh)
target_link_libraries(mps-rcs qtnh)

add_executable(mps-shor mps-shor.cpp)
add_dependencies(mps-shor qtnh)
target_link_libraries(mps-shor qtnh)

add_executable(permute permute.cpp)
add_dependencies(permute qtnh)
target_link_libraries(permute qtnh)

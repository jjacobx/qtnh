add_executable(rcs rcs.cpp)
add_dependencies(rcs qtnh)
target_link_libraries(rcs qtnh)

add_executable(riqft riqft.cpp)
add_dependencies(riqft qtnh)
target_link_libraries(riqft qtnh)

add_executable(rqft rqft.cpp)
add_dependencies(rqft qtnh)
target_link_libraries(rqft qtnh)
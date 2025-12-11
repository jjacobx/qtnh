add_executable(rcs rcs.cpp)
add_dependencies(rcs qtnh)
target_link_libraries(rcs qtnh)
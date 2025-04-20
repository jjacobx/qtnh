add_executable(qftn qftn.cpp)
add_dependencies(qftn qtnh)
target_link_libraries(qftn qtnh)

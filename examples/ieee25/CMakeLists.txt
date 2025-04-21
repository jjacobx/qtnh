add_executable(qftn qftn.cpp)
add_dependencies(qftn qtnh)
target_link_libraries(qftn qtnh)

add_executable(qft-acc qft-acc.cpp)
add_dependencies(qft-acc qtnh)
target_link_libraries(qft-acc qtnh)

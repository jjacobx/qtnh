add_executable(qftn qftn.cpp)
add_dependencies(qftn qtnh)
target_link_libraries(qftn qtnh)

add_executable(qftmps qftmps.cpp)
add_dependencies(qftmps qtnh)
target_link_libraries(qftmps qtnh)

add_executable(qftsv qftsv.cpp)
add_dependencies(qftsv qtnh)
target_link_libraries(qftsv qtnh)

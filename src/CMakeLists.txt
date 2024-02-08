add_library(qtnh
  env.cpp
  indexing.cpp
  dense-tensor.cpp
  tensor-network.cpp
)

target_compile_features(qtnh
  PUBLIC
  cxx_std_17
)

target_compile_options(qtnh
  PRIVATE
  -Wall
)

# Turn off Wsign-compare in Release builds
if (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(qtnh
    PRIVATE
    -Wno-sign-compare
  )
endif()

# Set DEBUG flag in Debug builds
if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(qtnh
    PRIVATE
    -DDEBUG
  )
endif()

target_include_directories(qtnh
  PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)


target_link_libraries(qtnh
  PUBLIC
  ${MATH_LIBRARY}
)

if(OpenMP_FOUND)
  target_link_libraries(qtnh
    PUBLIC
    OpenMP::OpenMP_CXX
  )
endif()

if(MPI_FOUND)
  target_link_libraries(qtnh
    PUBLIC
    MPI::MPI_CXX
  )
endif()

install(TARGETS qtnh
        EXPORT qtnh
)

install(FILES ${PROJECT_SOURCE_DIR}/include/tensor.hpp
  TYPE INCLUDE
)

# install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/qtnh 
#   TYPE INCLUDE
# )

install(EXPORT qtnh
  FILE qtnh.cmake
  NAMESPACE qtnh::
  DESTINATION lib/cmake
)

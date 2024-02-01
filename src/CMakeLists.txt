add_library(gtnh
  coords.cpp
  tensor.cpp
  dense-tensor.cpp
  env.cpp
)

target_compile_features(gtnh
  PUBLIC
  cxx_std_17
)

target_compile_options(gtnh
  PRIVATE
  -Wall
)

# Turn off Wsign-compare in Release builds
if (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(gtnh
    PRIVATE
    -Wno-sign-compare
  )
endif()

# Set DEBUG flag in Debug builds
if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(gtnh
    PRIVATE
    -DDEBUG
  )
endif()

target_include_directories(gtnh
  PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)


target_link_libraries(gtnh
  PUBLIC
  ${MATH_LIBRARY}
)

if(OpenMP_FOUND)
  target_link_libraries(gtnh
    PUBLIC
    OpenMP::OpenMP_CXX
  )
endif()

if(MPI_FOUND)
  target_link_libraries(gtnh
    PUBLIC
    MPI::MPI_CXX
  )
endif()

install(TARGETS gtnh
        EXPORT GTNH
)

install(FILES ${PROJECT_SOURCE_DIR}/include/tensor.hpp
  TYPE INCLUDE
)

# install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/gtnh 
#   TYPE INCLUDE
# )

install(EXPORT GTNH
  FILE GTNH.cmake
  NAMESPACE GTNH::
  DESTINATION lib/cmake
)

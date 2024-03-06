add_library(qtnh)

add_subdirectory(core)
add_subdirectory(tensor)

target_compile_features(qtnh
  PUBLIC
  cxx_std_17
)

target_compile_options(qtnh
  PRIVATE
  -Wall
  -Wdelete-non-virtual-dtor
)

# Turn off Wsign-compare in Release builds
if (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(qtnh
    PRIVATE
    -Wno-sign-compare
  )
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
  # Set DEBUG flag in Debug builds
  target_compile_options(qtnh
    PRIVATE
    -DDEBUG
  )

  # Additional Debug flags and libraries
  target_compile_options(qtnh
    PRIVATE
    -fsanitize=undefined,address
  )
  target_link_libraries(qtnh
    PUBLIC
    -fsanitize=undefined,address
  )
endif()

option(DEF_STENSOR_BCAST "Broadcast shared tensors from root." 1)
if (NOT DEF_STENSOR_BCAST)
  target_compile_options(qtnh
    PRIVATE
    -DDEF_STENSOR_BCAST=0
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

install(FILES ${PROJECT_SOURCE_DIR}/include/qtnh.hpp
  TYPE INCLUDE
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/core
  TYPE INCLUDE
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/tensor
  TYPE INCLUDE
)

install(EXPORT qtnh
  FILE qtnh.cmake
  NAMESPACE qtnh::
  DESTINATION lib/cmake
)

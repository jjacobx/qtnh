add_library(qtnh)

add_subdirectory(con)
add_subdirectory(core)
add_subdirectory(lalg)
add_subdirectory(tensor)

target_compile_features(qtnh
  PUBLIC
  cxx_std_17
)

target_compile_options(qtnh
  PRIVATE
  -Wall
  -Wextra
  #-Wshadow # - unfixable warnings
  -Wpedantic
  -Wdelete-non-virtual-dtor
  -Woverloaded-virtual
  -Wsuggest-override
  -Wconversion
  -Wno-sign-conversion
  -Wno-unused-parameter
)

# Set DEBUG flag in Debug builds
if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(qtnh
    PRIVATE
    -DDEBUG
  )
endif()

if (CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Optimisations set to O3.")
  target_compile_options(qtnh
    PRIVATE
    -O3
  )
endif()

option(MAP "Turn on ARM MAP profiling" OFF)
if (MAP)
  message(STATUS "ARM MAP profiling enabled.")
  set(FORGE_DIR "$ENV{FORGE_DIR}")
  target_link_options(qtnh
    PUBLIC
    -L${FORGE_DIR}/map/libs/default/gnu/ofi
    -lmap-sampler-pmpi
    -lmap-sampler
    -Wl,--eh-frame-hdr
    -Wl,-rpath=${FORGE_DIR}/map/libs/default/gnu/ofi
  )
  target_compile_options(qtnh
    PRIVATE
    -g
  )
endif()

option(FSANITIZE "Turn on fsanitize debug flags" OFF)
if (FSANITIZE)
  message(STATUS "Setting -fsanitize=undefined,address.")
  target_compile_options(qtnh
    PRIVATE
    -fsanitize=undefined,address
    -g
  )
  target_link_libraries(qtnh
    PUBLIC
    -fsanitize=undefined,address
    -lg
  )
endif()

option(AUTO_COMM_INIT "Automatically activate tensors." OFF)
if (AUTO_COMM_INIT)
  message(STATUS "Enabling automatic tensor activation.")
  target_compile_options(qtnh
    PRIVATE
    -DAUTO_COMM_INIT=1
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

if(MPI_FOUND)
  target_link_libraries(qtnh
    PUBLIC
    MPI::MPI_CXX
  )
endif()

if (ScaLAPACK_FOUND)
  message(STATUS "Linking ScaLAPACK.")
  target_link_libraries(qtnh
    PUBLIC
    scalapack
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

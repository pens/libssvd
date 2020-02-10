cmake_minimum_required(VERSION 3.0.0)
project(libssvd VERSION 0.1.0)

include_directories(/usr/local/magma/include /usr/local/cuda/include include)
file(GLOB SRC "src/*")

link_directories(/usr/local/magma/lib)

add_compile_options(-std=c++17 -g -pthread -Og -march=native -pipe)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_library(ssvd SHARED ${SRC})

target_link_libraries(ssvd blas lapack lapacke magma)
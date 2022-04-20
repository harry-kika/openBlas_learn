cmake_minimum_required(VERSION 3.19)
project(openBlas_learn)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-fPIC)

file(GLOB_RECURSE SRC_LIST source/*.cpp eval/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/openblas/include
        ${PROJECT_SOURCE_DIR}/openblas/include/openblas
        ${PROJECT_SOURCE_DIR}/source
        ${PROJECT_SOURCE_DIR}/eval)
link_directories(${PROJECT_SOURCE_DIR}/openblas/lib)

add_executable(openBlas_learn main.cpp ${SRC_LIST})

target_link_libraries(openBlas_learn
        PRIVATE openblas.a)

add_library(forPython SHARED forPython.cpp ${SRC_LIST})
set_target_properties(forPython PROPERTIES SUFFIX ".so" VERSIION 1.0 SOVERSIION 1)
target_link_libraries(forPython
        PRIVATE openblas.a)
cmake_minimum_required(VERSION 3.10)
project(MyFFTQAM)

set(CMAKE_CXX_COMPILER g++-15)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories(/opt/homebrew/include)
link_directories(/opt/homebrew/lib)

add_executable(main main.cpp)
target_link_libraries(main fftw3)
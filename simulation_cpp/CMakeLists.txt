cmake_minimum_required(VERSION 3.10)
project(MIMODetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_COMPILER g++-15)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    /opt/homebrew/include
)

link_directories(
    /opt/homebrew/lib
)

set(SOURCES
    include/src/modulation.cpp
    include/src/generation.cpp
    include/src/detection.cpp
    include/src/fourier.cpp
    main.cpp
)

add_executable(main ${SOURCES})
target_link_libraries(main fftw3f)
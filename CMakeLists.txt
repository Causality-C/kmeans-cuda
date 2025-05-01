cmake_minimum_required(VERSION 3.16)
project(KmeansCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If using CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 86)

# Add your sources
add_executable(KmeansCuda
    src/main.cpp
    src/config.cpp
    src/kmeans.cu
    src/profile.cpp
)

# Enable extended lambda for CUDA
target_compile_options(KmeansCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Include headers
target_include_directories(KmeansCuda PRIVATE include)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
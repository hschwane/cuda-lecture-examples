cmake_minimum_required(VERSION 3.16)

project(cudaTutorial CUDA CXX)

cmake_policy(SET CMP0104 OLD)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(1-ArrayCopy/C++)
add_subdirectory(1-ArrayCopy/CUDA)
add_subdirectory(2-MemoryManagement)
add_subdirectory(3-SharedMemory)
add_subdirectory(4-AtomicReduction)
add_subdirectory(5-CUB-Sort)
add_subdirectory(6-ThrustMemoryManagement)

cmake_minimum_required(VERSION 3.16)

add_executable(3-SharedMemory SharedMemory.cu)

target_compile_options(3-SharedMemory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)
set_target_properties(3-SharedMemory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

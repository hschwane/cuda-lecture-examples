cmake_minimum_required(VERSION 3.16)

add_executable(1-ArrayCopyCUDA ArrayCopy.cu)

target_compile_options(1-ArrayCopyCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)
set_target_properties(1-ArrayCopyCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

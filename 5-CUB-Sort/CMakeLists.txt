
add_executable(5-CUB-Sort CUB-Sort.cu)

target_compile_options(5-CUB-Sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)
set_target_properties(5-CUB-Sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

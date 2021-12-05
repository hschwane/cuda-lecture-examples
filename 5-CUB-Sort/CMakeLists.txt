
add_executable(5-CUB-Sort CUB-Sort.cu)

target_compile_options(5-CUB-Sort PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMPUTE_CAPABILITY}>)
set_target_properties(5-CUB-Sort PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

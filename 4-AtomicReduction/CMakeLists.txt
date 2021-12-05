
add_executable(4-AtomicReduction AtomicReduction.cu)

target_compile_options(4-AtomicReduction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)
set_target_properties(4-AtomicReduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

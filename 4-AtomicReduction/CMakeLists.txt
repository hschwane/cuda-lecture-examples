
add_executable(4-AtomicReduction AtomicReduction.cu)

target_compile_options(4-AtomicReduction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMPUTE_CAPABILITY}>)
set_target_properties(4-AtomicReduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(8a-SystemWideOptimizationNaive SystemWideOptimization.cu)

target_compile_options(8a-SystemWideOptimizationNaive PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
set_target_properties(8a-SystemWideOptimizationNaive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

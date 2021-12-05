
add_executable(8b-SystemWideOptimizationOptimized SystemWideOptimization.cu)

target_compile_options(8b-SystemWideOptimizationOptimized PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMPUTE_CAPABILITY}>)
set_target_properties(8b-SystemWideOptimizationOptimized PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

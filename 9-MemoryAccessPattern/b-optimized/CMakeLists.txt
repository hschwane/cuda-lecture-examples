
add_executable(9b-MemoryAccessPatternOptimized MemoryAccessPattern.cu)

target_compile_options(9b-MemoryAccessPatternOptimized PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
set_target_properties(9b-MemoryAccessPatternOptimized PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


add_executable(2-MemoryManagement ArrayCopy.cu)

target_compile_options(2-MemoryManagement PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
set_target_properties(2-MemoryManagement PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

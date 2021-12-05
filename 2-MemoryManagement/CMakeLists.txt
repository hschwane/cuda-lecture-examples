
add_executable(2-MemoryManagement ArrayCopy.cu)

target_compile_options(2-MemoryManagement PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)
set_target_properties(2-MemoryManagement PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

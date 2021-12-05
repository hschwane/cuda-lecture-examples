
add_executable(6-ThrustMemoryManagement ArrayCopy.cu)

target_compile_options(6-ThrustMemoryManagement PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_86,code=sm_86>)
set_target_properties(6-ThrustMemoryManagement PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

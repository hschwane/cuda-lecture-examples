
add_executable(6-ThrustMemoryManagement ArrayCopy.cu)

target_compile_options(6-ThrustMemoryManagement PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
set_target_properties(6-ThrustMemoryManagement PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

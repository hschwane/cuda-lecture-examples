
add_executable(3-SharedMemory SharedMemory.cu)

target_compile_options(3-SharedMemory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
set_target_properties(3-SharedMemory PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

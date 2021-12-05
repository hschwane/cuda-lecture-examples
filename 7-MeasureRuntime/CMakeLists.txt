
add_executable(7-MeasureRuntime ArrayCopy.cu)

target_compile_options(7-MeasureRuntime PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
set_target_properties(7-MeasureRuntime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

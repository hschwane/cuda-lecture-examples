
add_executable(1-ArrayCopyCUDA ArrayCopy.cu)

target_compile_options(1-ArrayCopyCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIONS}>)
set_target_properties(1-ArrayCopyCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

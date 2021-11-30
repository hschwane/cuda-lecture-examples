/*
 * cudaTutorial
 * ArrayCopy.cu
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#include <iostream>
#include <thrust/universal_vector.h>
#include "../shared/dataGenerator.h"
#include "../shared/cudaErrorCheck.h"

constexpr int numElements = 1e8;

void __global__ copyArray(const float* in, float* out, int num)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for(int i=id; i < num; i+=stride) {
        out[i] = in[i];
    }
}

int main () {
    thrust::cuda::universal_vector<float> input(numElements);
    thrust::cuda::universal_vector<float> output(numElements);
    genRandomData(input.begin(), input.end());

    copyArray<<<4096,256>>>( thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), numElements);
    gpuErrchk( cudaPeekAtLastError());
    gpuErrchk( cudaDeviceSynchronize());

    for(int i=0; i<10; i++)
        std::cout << output[i] << std::endl;

    return 0;
}
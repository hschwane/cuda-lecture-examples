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
#include <chrono>
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
    float* input;
    float* output;
    gpuErrchk( cudaMallocManaged(&input, numElements*sizeof(float)));
    gpuErrchk( cudaMallocManaged(&output, numElements*sizeof(float)));

    genRandomData(input, input+numElements);

    copyArray<<<4096,256>>>(input, output, numElements);
    gpuErrchk( cudaPeekAtLastError());
    gpuErrchk( cudaDeviceSynchronize());

    int numBenchmarks = 10;
    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<numBenchmarks; i++)
    {
        copyArray<<<4096,256>>>(input, output, numElements);
        gpuErrchk( cudaPeekAtLastError());
        gpuErrchk( cudaDeviceSynchronize());
    }
    auto duration = std::chrono::steady_clock::now() - start;
    std::cout << "Kernel took "
              << std::chrono::duration_cast< std::chrono::duration <float >>( duration ).count() * 1000 / numBenchmarks
              << "ms" << std::endl;

    for(int i=0; i<10; i++)
        std::cout << output[i] << std::endl;

    cudaFree(input);
    cudaFree(output);
    return 0;
}
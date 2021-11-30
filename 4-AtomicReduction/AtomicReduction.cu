/*
 * cudaTutorial
 * SharedMemory.cu
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#include <iostream>
#include "../shared/dataGenerator.h"
#include "../shared/cudaErrorCheck.h"

constexpr int numElements = 1e8;

void __global__ reduceAdd(const float* in, float* out, int num)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    float localSum = 0.0f;
    for(int i=id; i < num; i+=stride) {
        localSum += in[i];
    }
    atomicAdd(out,localSum);
}

int main () {
    float* input;
    float* output;
    gpuErrchk( cudaMallocManaged(&input, numElements*sizeof(float)));
    gpuErrchk( cudaMallocManaged(&output, 1*sizeof(float)));

    genRandomData(input, input+numElements);
    *output = 0.0f;

    reduceAdd<<<4096,128>>>(input, output, numElements);
    gpuErrchk( cudaPeekAtLastError());
    gpuErrchk( cudaDeviceSynchronize());

    std::cout << *output << std::endl;

    cudaFree(input);
    cudaFree(output);
    return 0;
}
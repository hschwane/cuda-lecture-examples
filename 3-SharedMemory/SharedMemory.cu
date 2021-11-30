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
constexpr int blockSize = 128;

void __global__ addListToArray(const float* in, float* out, int num, const float* valuesToAdd)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // load data from global to shared memory
    __shared__ float sharedData[blockSize];
    sharedData[threadIdx.x] = valuesToAdd[threadIdx.x];

    // sync, to make sure all threads are done writing date
    __syncthreads();

    // gird stride loop, to allow multiple items to be processed per thread
    for(int i=id; i < num; i+=stride) {

        // load input from global memory, add all elements from the shared memory
        float sum = in[i];
        for(int j = 0; j<blockSize; j++) {
            sum += sharedData[j];
        }
        out[i] = sum;
    }
}


int main () {
    float* input;
    float* valuesToAdd;
    float* output;
    gpuErrchk( cudaMallocManaged(&input, numElements*sizeof(float)));
    gpuErrchk( cudaMallocManaged(&output, numElements*sizeof(float)));
    gpuErrchk( cudaMallocManaged(&valuesToAdd, blockSize));

    genRandomData(input, input+numElements);
    genRandomData(valuesToAdd, valuesToAdd+blockSize);

    addListToArray<<<1e5,blockSize>>>(input, output, numElements, valuesToAdd);
    gpuErrchk( cudaPeekAtLastError());
    gpuErrchk( cudaDeviceSynchronize());

    for(int i=0; i<10; i++)
        std::cout << output[i] << std::endl;

    cudaFree(input);
    cudaFree(output);
    return 0;
}


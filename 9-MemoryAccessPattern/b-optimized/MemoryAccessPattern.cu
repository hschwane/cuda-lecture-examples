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
#include "../../shared/dataGenerator.h"
#include "../../shared/cudaErrorCheck.h"

constexpr int width = 16384;
constexpr int height = 16384;

void __global__ matrixAddition(const float* inA, const float* inB, float* out, int w, int h)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int idy = threadIdx.y+blockDim.y*blockIdx.y;

    if(idx < w && idy < h) {
        int id = idy * w + idx; // same as using array[idy][idx] on a 2D Array
        out[id] = inA[id] + inB[id];
    }
}


int main ()
{
    int numElements = width*height;

    float* inputA;
    float* inputB;
    float* output;
    gpuErrchk( cudaMallocManaged(&inputA, numElements*sizeof(float)));
    gpuErrchk( cudaMallocManaged(&inputB, numElements*sizeof(float)));
    gpuErrchk( cudaMallocManaged(&output, numElements*sizeof(float)));

    genRandomData(inputA, inputA+numElements);
    genRandomData(inputB, inputB+numElements);

    dim3 threads(32,32);
    dim3 blocks((width+threads.x-1)/threads.x, (height+threads.y-1)/threads.y);

    matrixAddition<<<blocks,threads>>>(inputA, inputB, output, width, height);
    gpuErrchk( cudaPeekAtLastError());
    gpuErrchk( cudaDeviceSynchronize());

    for(int i=0; i<10; i++)
        std::cout << output[i] << std::endl;

    cudaFree(inputA);
    cudaFree(inputB);
    cudaFree(output);
    return 0;
}
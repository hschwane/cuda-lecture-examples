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
    std::vector<float> input_h(numElements);
    std::vector<float> output_h(numElements);
    genRandomData(input_h.begin(), input_h.end());

    float* input_d;
    float* output_d;
    gpuErrchk( cudaMalloc(&input_d, numElements*sizeof(float)));
    gpuErrchk( cudaMalloc(&output_d, numElements*sizeof(float)));

    gpuErrchk( cudaMemcpy(input_d, input_h.data(), input_h.size()*sizeof(float), cudaMemcpyHostToDevice));

    copyArray<<<4096,256>>>(input_d, output_d, numElements);
    gpuErrchk(cudaPeekAtLastError());
    // cudaDeviceSynchronize();
    // not needed here, cudaMemcopy will synchronize automatically

    gpuErrchk( cudaMemcpy(output_h.data(), output_d, input_h.size()*sizeof(float), cudaMemcpyDeviceToHost));

    for(int i=0; i<10; i++)
        std::cout << output_h[i] << std::endl;

    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}
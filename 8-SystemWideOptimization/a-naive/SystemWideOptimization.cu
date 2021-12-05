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

constexpr int numElements = 1e8;

// dummy kernel to simulate some work being done on all elements
void __global__ dummyKernel(const float* in, float* out, int num)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for(int i=id; i < num; i+=stride) {

        float f = in[i];
        for(int j=0; j<10; j++) {
            float g1 = sin(in[i]) * tanh(in[i]) + cos(in[i]) * tan(in[i]);
            float g2 = sin(in[i]) * cos(in[i]) + tanh(in[i]) * tan(in[i]);
            f = g1 * g1 * g2 * g2;
            f = sqrt(f);
        }
        out[i] = exp(f);
    }
}

// dummy function to simulate generating input data
void generateInput(std::vector<float>& data) {
    // generating all the random numbers on the cpu is way too slow,
    // so we just use the same data over and over again
}

// dummy function to simulate CPU processing the data
void processOutput(const std::vector<float>& data)
{
    float sum = 0;
    for(int i=0; i<10 && i<data.size(); i+=10) {
        sum += data[i];
    }
    std::cout << sum << std::endl;
}

int main ()
{
    std::vector<float> input_h(numElements);
    std::vector<float> output_h(numElements);
    genRandomData(input_h.begin(), input_h.end());

    float* input_d;
    float* output_d;
    gpuErrchk( cudaMalloc(&input_d, numElements*sizeof(float)));
    gpuErrchk( cudaMalloc(&output_d, numElements*sizeof(float)));

    for(int i = 0; i < 10; i++) {
        generateInput(input_h);
        gpuErrchk( cudaMemcpy(input_d, input_h.data(), input_h.size()*sizeof(float), cudaMemcpyHostToDevice));

        // simulate some work on the data
        for(int j=0; j<6; j++) {
            dummyKernel<<<4096, 256>>>(input_d, output_d, numElements);
            gpuErrchk(cudaPeekAtLastError());
        }

        gpuErrchk( cudaMemcpy(output_h.data(), output_d, input_h.size()*sizeof(float), cudaMemcpyDeviceToHost));
        processOutput(output_h);
    }


    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}
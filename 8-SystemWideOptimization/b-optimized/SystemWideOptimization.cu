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
void generateInput(float* data) {
    // generating all the random numbers on the cpu is way too slow,
    // so we just use the same data over and over again
}

// dummy function to simulate CPU processing the data
void processOutput(const float* data, int numData)
{
    float sum = 0;
    for(int i=0; i<10 && i<numData; i+=10) {
        sum += data[i];
    }
    std::cout << sum << std::endl;
}

int main ()
{
    // to do async memcopy we need to use pinned memory on the CPU
    float* input_h;
    float* output_h;
    gpuErrchk( cudaMallocHost(&input_h, numElements*sizeof(float)));
    gpuErrchk( cudaMallocHost(&output_h, numElements*sizeof(float)));

    genRandomData(input_h, input_h+numElements);

    // create 2 pair of buffers,
    // one pair to run cuda kernel on and one pair to perform copy
    float* input_d_processing;
    float* output_d_processing;
    float* input_d_copy;
    float* output_d_copy;
    gpuErrchk( cudaMalloc(&input_d_processing, numElements*sizeof(float)));
    gpuErrchk( cudaMalloc(&output_d_processing, numElements*sizeof(float)));
    gpuErrchk( cudaMalloc(&input_d_copy, numElements*sizeof(float)));
    gpuErrchk( cudaMalloc(&output_d_copy, numElements*sizeof(float)));

    // create cuda streams to overlap execution and data transfer
    cudaStream_t s1; // run kernel in s1
    cudaStream_t s2; // run upload in s2
    cudaStream_t s3; // run download in s3
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);

    // generate and copy the first data
    generateInput(input_h);
    gpuErrchk(cudaMemcpy(input_d_processing, input_h, numElements * sizeof(float), cudaMemcpyHostToDevice));

    int numRuns=10;
    for(int i = 0; i < numRuns; i++) {

        // simulate some work on the data
        // kernels are launched async in stream s1
        for(int j=0; j<6; j++) {
            // launch parameters: <<< numBlocks, blockSize, Shared Memory, Stream >>>
            dummyKernel<<<4096, 256, 0, s1>>>(input_d_processing, output_d_processing, numElements);
            gpuErrchk(cudaPeekAtLastError());
        }

        // download and process __previous__ output data, except for first run
        if(i>0) {
            if(i>1) {
                // if we have local data already, make sure we process them before overriding
                // if processing data is slow, we could overlap that as well, using another buffer
                processOutput(output_h, numElements);
            }
            // now download the previous data async using stream s2
            gpuErrchk(cudaMemcpyAsync(output_h, output_d_copy, numElements * sizeof(float), cudaMemcpyDeviceToHost, s3));
        }

        // prepare __next__ input data, we upload it async using stream s4
        if(i<numRuns-1) {
            // if generating data were very slow, we could overlap that as well using another buffer
            generateInput(input_h);
            gpuErrchk(cudaMemcpyAsync(input_d_copy, input_h, numElements * sizeof(float), cudaMemcpyHostToDevice, s2));
        }

        // sync all streams, so we know gpu and cpu are done working and swapping buffers is safe
        // use cuda "events" for more control over synchronization between gpu and cpu
        gpuErrchk(cudaDeviceSynchronize());

        // swap buffers
        std::swap(input_d_copy, input_d_processing);
        std::swap(output_d_copy, output_d_processing);
    }

    // download and process final set of data
    processOutput(output_h, numElements);
    gpuErrchk(cudaMemcpy(output_h, output_d_copy, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    processOutput(output_h, numElements);

    cudaFree(input_d_processing);
    cudaFree(output_d_processing);
    cudaFree(input_d_copy);
    cudaFree(output_d_copy);

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);
    return 0;
}
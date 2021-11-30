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
#include <cub/cub.cuh>
#include "../shared/dataGenerator.h"
#include "../shared/cudaErrorCheck.h"

constexpr int numElements = 1e8;

int main () {
    float* input;
    float* output;
    gpuErrchk( cudaMallocManaged(&input, numElements*sizeof(float)));
    gpuErrchk( cudaMallocManaged(&output, numElements*sizeof(float)));

    genRandomData(input, input+numElements);

    // Determine temporary device storage requirements
    // this is done by passing 0 as the temp storage
    void     *tempStorage_d = NULL;
    size_t   tempStorageSize = 0;
    gpuErrchk( cub::DeviceRadixSort::SortKeys(tempStorage_d, tempStorageSize, input, output, numElements));

    // Allocate temporary storage
    gpuErrchk( cudaMalloc(&tempStorage_d, tempStorageSize));

    // Run sorting operation
    gpuErrchk( cub::DeviceRadixSort::SortKeys(tempStorage_d, tempStorageSize, input, output, numElements));

    gpuErrchk( cudaDeviceSynchronize());
    for(int i=0; i<10; i++)
        std::cout << output[i] << std::endl;

    cudaFree(input);
    cudaFree(output);
    return 0;
}
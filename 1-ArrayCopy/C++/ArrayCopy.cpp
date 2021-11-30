/*
 * cudaTutorial
 * ArrayCopy.cpp
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */

#include <iostream>
#include <vector>
#include "../../shared/dataGenerator.h"

constexpr int numElements = 1e8;

void copyArray(const std::vector<float>& in, std::vector<float>& out)
{
    for(int i = 0; i < in.size(); i++) {
        out[i] = in[i];
    }
}

int main () {
    std::vector<float> input(numElements);
    std::vector<float> output(numElements);
    genRandomData(input.begin(), input.end());

    copyArray(input,output);

    for(int i=0; i<10; i++)
        std::cout << output[i] << std::endl;

    return 0;
}

/*
 * cudaTutorial
 * helper.h
 *
 * @author: Hendrik Schwanekamp
 * @mail:   hendrik.schwanekamp@gmx.net
 *
 * Copyright (c) 2021 Hendrik Schwanekamp
 *
 */
#ifndef CUDATUTORIAL_DATAGENERATOR_H
#define CUDATUTORIAL_DATAGENERATOR_H

#include "../../../../../usr/include/c++/9/random"

template<typename itT>
void genRandomData(itT begin, itT end){
    std::random_device seed;
    std::default_random_engine rng(seed());
    std::uniform_real_distribution<float> dist(0,100);
    for(auto it=begin; it != end; it++ ) {
        *it = dist(rng);
    }
}

#endif //CUDATUTORIAL_DATAGENERATOR_H

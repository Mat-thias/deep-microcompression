#ifndef FUSED_LAYERS_H
#define FUSED_LAYERS_H

#include "layer.h"
#include "conv.h"
#include "linear.h"
#include "pad.h"


// inline float relu(float val) { return (val < 0) ? 0 : val;}
// inline int32_t relu(int32_t val) { return (val < 0) ? 0 : val;}

// inline float relux(float val, float x) { return (val < 0) ? 0 : (val > x) ? x : val;}
// inline int32_t relux(int32_t val, int32_t x) { return (val < 0) ? 0 : (val > x) ? x : val;}

class Conv2dReLU: public Conv2d {
public:
    using Conv2d::Conv2d;

    void forward(float* input, float* output);
    void forward(int8_t* input, int8_t* output);
};


class LinearReLU: public Linear {
public:

    using Linear::Linear;

    void forward(float* input, float* output);
    void forward(int8_t* input, int8_t* output);
};

class Conv2dReLU6: public Conv2d {
public:
    using Conv2d::Conv2d;

    void forward(float* input, float* output);
    void forward(int8_t* input, int8_t* output);
};


class LinearReLU6: public Linear {
public:

    using Linear::Linear;

    void forward(float* input, float* output);
    void forward(int8_t* input, int8_t* output);
};

#endif// FUSED_LAYERS_H
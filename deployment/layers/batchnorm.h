#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "layer.h"

class BatchNorm2d : public Layer{

private:
    // Input tensor dimensions
    uint32_t input_channel_size;  ///< Number of input channels
    uint32_t input_row_size;      ///< Height of input feature map
    uint32_t input_col_size;      ///< Width of input feature map

    // Weight and bias tensors
    const float* weight;         ///< Pointer to weight tensor
    const float* bias;           ///< Pointer to bias tensor
    const float* running_mean;
    const float* running_var;
    

public:
    BatchNorm2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
                const float* weight, const float* bias, const float* running_mean, const float* running_var);

    void forward(float* input, float* output);
};

#endif // BATCHNORM_H
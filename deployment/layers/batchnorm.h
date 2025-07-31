#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "layer.h"


#ifdef STATIC_QUANTIZATION_PER_TENSOR // QUANTIZATION_TYPE


#else // DYNAMIC_QUANTIZATION_PER_TENSOR

class BatchNorm2d : public Layer{

private:
    // Input tensor dimensions
    uint32_t input_channel_size;  ///< Number of input channels
    uint32_t input_row_size;      ///< Height of input feature map
    uint32_t input_col_size;      ///< Width of input feature map

    // Weight and bias tensors
    const float* folded_weight;         ///< Pointer to weight tensor
    const float* folded_bias;           ///< Pointer to bias tensor    

public:
    BatchNorm2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
                const float* folded_weight, const float* folded_bias);

    void forward(float* input, float* output);
};


#endif // QUANTIZATION_TYPE



#endif // BATCHNORM_H
/**
 * @file layer.h
 * @brief Base layer interface with support for:
 *       1. Non-quantized models (float)
 *       2. Static quantized models per tensor (int8_t)
 */

#ifndef LAYER_H
#define LAYER_H

#include <stdint.h>  // For int8_t type
#include <float.h>   // For floating-point constants
#include <math.h>    // For math operations


struct Padding_t {
    uint8_t padding_left;
    uint8_t padding_right;
    uint8_t padding_top;
    uint8_t padding_bottom;

    bool is_padded() {
        return (this->padding_bottom + this->padding_top + 
                this->padding_left + this->padding_right) > 0;
    }
};


#if !defined(STATIC_QUANTIZATION_PER_TENSOR)

/**
 * @class Layer
 * @brief Abstract base class for all floating-point layers
 * 
 * Provides the interface for forward propagation in non-quantized networks.
 * All concrete layer types must implement the forward() method.
 */
class Layer {
public:
    /**
     * @brief Forward pass interface for floating-point layers
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     * 
     * @note Pure virtual function - must be implemented by derived classes
     */
    virtual void forward(float* input, float* output) = 0;
};

#else // STATIC_QUANTIZATION_PER_TENSOR

/**
 * @class Layer
 * @brief Abstract base class for all quantized layers
 * 
 * Provides the interface for forward propagation in quantized networks.
 * All concrete layer types must implement the forward() method.
 */
class Layer {
public:
    /**
     * @brief Forward pass interface for quantized layers
     * @param input Pointer to quantized input tensor (int8_t)
     * @param output Pointer to quantized output tensor (int8_t)
     * 
     * @note Pure virtual function - must be implemented by derived classes
     */
    virtual void forward(int8_t* input, int8_t* output) = 0;
};

#endif // STATIC_QUANTIZATION_PER_TENSOR

#endif // LAYER_H
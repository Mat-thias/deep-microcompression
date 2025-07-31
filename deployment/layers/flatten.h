/**
 * @file flatten.h
 * @brief Header for Flatten layer with support for:
 *      1. Non-quantized models (float)
 *      2. Static quantized models per tensor (int8_t)
 */

#ifndef FLATTEN_H
#define FLATTEN_H

#include "layer.h"



#ifdef STATIC_QUANTIZATION_PER_TENSOR // QUANTIZATION_TYPE


class Flatten : public Layer {
private:
    uint32_t input_size;  ///< Total number of elements in input tensor

public:
    /**
     * @brief Constructor for quantized Flatten layer
     * @param size Number of elements in input tensor
     */
    Flatten(uint32_t size);

    /**
     * @brief Forward pass for quantized flatten operation
     * @param input Pointer to input tensor (int8_t)
     * @param output Pointer to output tensor (int8_t)
     */
    void forward(int8_t* input, int8_t* output);
};

#else // DYNAMIC_QUANTIZATION_PER_TENSOR

/**
 * @class Flatten
 * @brief Flatten layer implementation for floating-point models
 * 
 * This layer reshapes multi-dimensional input into 1D output without
 * modifying values. Used for transition between convolutional and dense layers.
 */
class Flatten : public Layer {
private:
    uint32_t input_size;  ///< Total number of elements in input tensor

public:
    /**
     * @brief Constructor for floating-point Flatten layer
     * @param size Number of elements in input tensor
     */
    Flatten(uint32_t size);

    /**
     * @brief Forward pass for floating-point flatten operation
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    void forward(float* input, float* output);
};



#endif // QUANTIZATION_TYPE

// #if !defined(STATIC_QUANTIZATION_PER_TENSOR)


// #else // STATIC_QUANTIZATION_PER_TENSOR

// /**
//  * @class Flatten
//  * @brief Flatten layer implementation for quantized models
//  * 
//  * Quantized version maintains same functionality as floating-point version,
//  * but operates on int8_t tensors.
//  */
// class Flatten : public Layer {
// private:
//     uint32_t input_size;  ///< Total number of elements in input tensor

// public:
//     /**
//      * @brief Constructor for quantized Flatten layer
//      * @param size Number of elements in input tensor
//      */
//     Flatten(uint32_t size);

//     /**
//      * @brief Forward pass for quantized flatten operation
//      * @param input Pointer to input tensor (int8_t)
//      * @param output Pointer to output tensor (int8_t)
//      */
//     void forward(int8_t* input, int8_t* output);
// };

// #endif // STATIC_QUANTIZATION_PER_TENSOR

#endif // FLATTEN_H
/**
 * @file flatten.cpp
 * @brief Implementation of Flatten layer with support for:
 *      1. None quantized model (float)
 *      2. Static quantized model per tensor (int8_t)
 */

#include "flatten.h"




#ifdef STATIC_QUANTIZATION_PER_TENSOR // QUANTIZATION_TYPE

Flatten::Flatten(uint32_t input_size) {
    this->input_size = input_size;
}

void Flatten::forward(int8_t* input, int8_t* output) {
    // Perform element-wise copy (no transformation needed)
    for (uint32_t i = 0; i < this->input_size; i++) {
        output[i] = input[i];
    }
}

#else // DYNAMIC_QUANTIZATION_PER_TENSOR

/**
 * @brief Constructor for floating-point Flatten layer
 * @param input_size Number of elements in input tensor
 * 
 * @note Flatten operation doesn't modify values, just reshapes the tensor
 */
Flatten::Flatten(uint32_t input_size) {
    this->input_size = input_size;
}

/**
 * @brief Forward pass for floating-point Flatten
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * Simply copies input to output as flattening is just a view operation.
 * Maintains same memory layout but changes tensor shape interpretation.
 */
void Flatten::forward(float* input, float* output) {
    // Perform element-wise copy (no transformation needed)
    for (uint32_t i = 0; i < this->input_size; i++) {
        output[i] = input[i];
    }
}



#endif // QUANTIZATION_TYPE

// #if !defined(STATIC_QUANTIZATION_PER_TENSOR)

// #else // STATIC_QUANTIZATION_PER_TENSOR

// /**
//  * @brief Constructor for quantized Flatten layer
//  * @param input_size Number of elements in input tensor
//  * 
//  * @note Quantized version works with int8_t values but maintains same
//  *       behavior as floating-point version
//  */
// Flatten::Flatten(uint32_t input_size) {
//     this->input_size = input_size;
// }

// /**
//  * @brief Forward pass for quantized Flatten
//  * @param input Pointer to quantized input tensor (int8_t)
//  * @param output Pointer to quantized output tensor (int8_t)
//  * 
//  * Performs same operation as floating-point version but with int8_t values.
//  * No quantization parameters needed as values are just copied.
//  */
// void Flatten::forward(int8_t* input, int8_t* output) {
//     // Perform element-wise copy (no transformation needed)
//     for (uint32_t i = 0; i < this->input_size; i++) {
//         output[i] = input[i];
//     }
// }

// #endif // STATIC_QUANTIZATION_PER_TENSOR
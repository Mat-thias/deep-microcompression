/**
 * @file layer.cpp
 * @brief Base layer implementation with support for:
 *      1. Non-quantized models (float)
 *      2. Static quantized models per tensor (int8_t)
 */

#include "layer.h"

#ifdef STATIC_QUANTIZATION_PER_TENSOR // QUANTIZATION_TYPE

void Layer::forward(int8_t* input, int8_t* output) {
    // Intentionally empty - to be implemented by derived classes
}

#else // DYNAMIC_QUANTIZATION_PER_TENSOR


/**
 * @brief Default forward pass implementation for floating-point layers
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * @note This base implementation does nothing and should be overridden
 *       by derived layer classes.
 */
void Layer::forward(float* input, float* output) {
    // Intentionally empty - to be implemented by derived classes
}


#endif // QUANTIZATION_TYPE

// #if !defined(STATIC_QUANTIZATION_PER_TENSOR)

// #else // STATIC_QUANTIZATION_PER_TENSOR

// /**
//  * @brief Default forward pass implementation for quantized layers
//  * @param input Pointer to quantized input tensor (int8_t)
//  * @param output Pointer to quantized output tensor (int8_t)
//  * 
//  * @note This base implementation does nothing and should be overridden
//  *       by derived layer classes in quantized models.
//  */
// void Layer::forward(int8_t* input, int8_t* output) {
//     // Intentionally empty - to be implemented by derived classes
// }

// #endif // STATIC_QUANTIZATION_PER_TENSOR



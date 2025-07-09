/**
 * @file activation.cpp
 * @brief Implementation of ReLU activation layer with support:
 *      1. None quantized model
 *      2. Dynamic quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 *      3. Static quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 */

#include "activation.h"

#if !defined(STATIC_QUANTIZATION_PER_TENSOR)

/**
 * @brief Constructor for floating-point ReLU layer
 * @param input_size Number of elements in input tensor
 */
ReLU::ReLU(uint32_t input_size) {
    this->input_size = input_size;
}

/**
 * @brief Forward pass for floating-point ReLU
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * Computes: output[i] = max(0, input[i]) for each element
 */
void ReLU::forward(float* input, float* output) {
    // Apply ReLU function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        output[i] = (input[i] > 0) ? input[i] : 0;
    }
}



ReLU6::ReLU6(uint32_t input_size) {
    this->input_size = input_size;
}

void ReLU6::forward(float* input, float* output) {
    // Apply ReLU6 function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        output[i] = (input[i] < 0) ? 0 : (input[i] > 6) ? 6 : input[i];
    }
}

#else // STATIC_QUANTIZATION_PER_TENSOR

/**
 * @brief Constructor for quantized ReLU layer
 * @param input_size Number of elements in input tensor
 * @param input_zero_point Zero-point for quantized input
 */
ReLU::ReLU(uint32_t input_size, int8_t input_zero_point) {
    this->input_size = input_size;
    this->input_zero_point = input_zero_point;
}

/**
 * @brief Forward pass for quantized ReLU
 * @param input Pointer to quantized input tensor (int8_t)
 * @param output Pointer to quantized output tensor (int8_t)
 * 
 * Computes: output[i] = max(input_zero_point, input[i]) for each element
 * Note: Zero-point acts as the "zero" threshold in quantized space
 */
void ReLU::forward(int8_t* input, int8_t* output) {
    // Apply quantized ReLU function element-wise
    for (uint32_t i = 0; i < this->input_size; i++) {
        output[i] = (input[i] > this->input_zero_point) ? input[i] : this->input_zero_point;
    }
}

#endif // STATIC_QUANTIZATION_PER_TENSOR
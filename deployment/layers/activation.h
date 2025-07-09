/**
 * @file activation.h
 * @brief Header for ReLU activation layer with support:
 *      1. None quantized model
 *      2. Dynamic quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 *      3. Static quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 * 
 * The implementation switches between modes based on STATIC_QUANTIZATION_PER_TENSOR:
 * - Floating-point mode: Operates on float tensors
 * - Static Quantized mode: Operates on int8_t tensors with zero-point
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layer.h"

#if !defined(STATIC_QUANTIZATION_PER_TENSOR)

/**
 * @brief Floating-point ReLU activation layer
 * 
 * Implements standard ReLU: output = max(0, input)
 */
class ReLU : public Layer {
private:
    uint32_t input_size;  ///< Number of elements in input tensor
    
public:
    /**
     * @brief Constructor for floating-point ReLU
     * @param input_size Number of elements in input tensor
     */
    ReLU(uint32_t input_size);

    /**
     * @brief Forward pass for floating-point ReLU
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    void forward(float* input, float* output);
};



class ReLU6 : public Layer {
private:
    uint32_t input_size;  ///< Number of elements in input tensor
    
public:
    /**
     * @brief Constructor for floating-point ReLU
     * @param input_size Number of elements in input tensor
     */
    ReLU6(uint32_t input_size);

    /**
     * @brief Forward pass for floating-point ReLU
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    void forward(float* input, float* output);
};

#else // STATIC_QUANTIZATION_PER_TENSOR

/**
 * @brief Quantized ReLU activation layer
 * 
 * Implements quantized ReLU: output = max(zero_point, input)
 */
class ReLU : public Layer {
private:
    uint32_t input_size;      ///< Number of elements in input tensor
    int8_t input_zero_point;  ///< Zero-point for quantized input
    
public:
    /**
     * @brief Constructor for quantized ReLU
     * @param input_size Number of elements in input tensor
     * @param input_zero_point Zero-point for quantized input
     */
    ReLU(uint32_t input_size, int8_t input_zero_point);

    /**
     * @brief Forward pass for quantized ReLU
     * @param input Pointer to input tensor (int8_t)
     * @param output Pointer to output tensor (int8_t)
     */
    void forward(int8_t* input, int8_t* output);
};

#endif // STATIC_QUANTIZATION_PER_TENSOR

#endif // ACTIVATION_H
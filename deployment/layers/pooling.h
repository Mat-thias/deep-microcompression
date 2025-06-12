/**
 * @file pooling.h
 * @brief MaxPool2d layer definition for 2D max pooling operations.
 * 
 * Supports both floating-point and quantized (int8_t) inference modes.
 */

#ifndef POOLING_H
#define POOLING_H

#include "layer.h"

// Conditional compilation for quantization support
#if !defined(STATIC_QUANTIZATION_PER_TENSOR)

/**
 * @brief MaxPool2d layer for floating-point inference.
 * 
 * Performs 2D max pooling operation on float input tensors.
 */
class MaxPool2d : public Layer {
private:
    uint32_t input_channel_size;  ///< Number of input channels
    uint32_t input_row_size;      ///< Height of input feature map
    uint32_t input_col_size;      ///< Width of input feature map

    uint32_t output_row_size;     ///< Height of output feature map
    uint32_t output_col_size;     ///< Width of output feature map

    uint32_t kernel_size;         ///< Size of pooling window (square)
    uint32_t stride;              ///< Stride for pooling operation
    uint32_t padding;             ///< Padding size around input

public:
    /**
     * @brief Constructor for floating-point MaxPool2d layer.
     * 
     * @param input_channel_size Number of input channels
     * @param input_row_size Height of input feature map
     * @param input_col_size Width of input feature map
     * @param kernel_size Size of pooling window
     * @param stride Stride for pooling operation
     * @param padding Padding size around input
     */
    MaxPool2d(uint32_t input_channel_size, 
              uint32_t input_row_size, 
              uint32_t input_col_size,
              uint32_t kernel_size, 
              uint32_t stride, 
              uint32_t padding);

    /**
     * @brief Forward pass for floating-point max pooling.
     * 
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    void forward(float* input, float* output);
};

#else // STATIC_QUANTIZATION_PER_TENSOR

/**
 * @brief MaxPool2d layer for quantized (int8_t) inference.
 * 
 * Performs 2D max pooling operation on quantized input tensors.
 */
class MaxPool2d : public Layer {
private:
    uint32_t input_channel_size;  ///< Number of input channels
    uint32_t input_row_size;      ///< Height of input feature map
    uint32_t input_col_size;      ///< Width of input feature map

    uint32_t output_row_size;     ///< Height of output feature map
    uint32_t output_col_size;     ///< Width of output feature map

    uint32_t kernel_size;         ///< Size of pooling window (square)
    uint32_t stride;              ///< Stride for pooling operation
    uint32_t padding;             ///< Padding size around input

public:
    /**
     * @brief Constructor for quantized MaxPool2d layer.
     * 
     * @param input_channel_size Number of input channels
     * @param input_row_size Height of input feature map
     * @param input_col_size Width of input feature map
     * @param kernel_size Size of pooling window
     * @param stride Stride for pooling operation
     * @param padding Padding size around input
     */
    MaxPool2d(uint32_t input_channel_size, 
              uint32_t input_row_size, 
              uint32_t input_col_size,
              uint32_t kernel_size, 
              uint32_t stride, 
              uint32_t padding);

    /**
     * @brief Forward pass for quantized max pooling.
     * 
     * @param input Pointer to input tensor (int8_t)
     * @param output Pointer to output tensor (int8_t)
     */
    void forward(int8_t* input, int8_t* output);
};

#endif // STATIC_QUANTIZATION_PER_TENSOR

#endif // POOLING_H
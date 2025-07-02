/**
 * @file conv.h
 * @brief Header for 2D convolution layer with support or:
 *      1. None quantized model
 *      2. Dynamic quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 *      3. Static quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 * 
 * The implementation is selected via compile-time definitions:
 * - QUANTIZATION_NONE: Floating-point
 * - DYNAMIC_QUANTIZATION_PER_TENSOR: Dynamic quantization
 * - STATIC_QUANTIZATION_PER_TENSOR: Static quantization
 * 
 * - QUANTIZATION_BITWIDTH: 8, 4
 */

#ifndef CONV_H
#define CONV_H

#include "layer.h"

#if defined(QUANTIZATION_NONE) || (!defined(DYNAMIC_QUANTIZATION_PER_TENSOR) && !defined(DYNAMIC_QUANTIZATION_PER_CHANNEL) \
                               && !defined(STATIC_QUANTIZATION_PER_TENSOR) && !defined(STATIC_QUANTIZATION_PER_CHANNEL))

// ======================================================================
// Floating-Point Conv2d
// ======================================================================

/**
 * @brief Floating-point 2D convolution layer
 */
class Conv2d : public Layer {
private:
    // Input tensor dimensions
    uint32_t input_channel_size;  ///< Number of input channels
    uint32_t input_row_size;      ///< Height of input feature map
    uint32_t input_col_size;      ///< Width of input feature map

    // Output tensor dimensions
    uint32_t output_channel_size; ///< Number of output channels
    uint32_t output_row_size;     ///< Height of output feature map
    uint32_t output_col_size;     ///< Width of output feature map

    // Kernel parameters
    uint32_t kernel_row_size;     ///< Height of convolution kernel
    uint32_t kernel_col_size;     ///< Width of convolution kernel

    // Operation parameters
    uint32_t stride_row;         ///< Vertical stride
    uint32_t stride_col;         ///< Horizontal stride
    uint32_t padding;            ///< Padding type (0=VALID, 1=SAME)
    uint32_t groups;

    // Weight and bias tensors
    const float* weight;         ///< Pointer to weight tensor
    const float* bias;           ///< Pointer to bias tensor

public:
    /**
     * @brief Constructor for floating-point Conv2d
     * @param input_channel_size Number of input channels
     * @param input_row_size Input height in pixels
     * @param input_col_size Input width in pixels
     * @param output_channel_size Number of output channels
     * @param kernel_row_size Kernel height
     * @param kernel_col_size Kernel width
     * @param stride_row Vertical stride
     * @param stride_col Horizontal stride
     * @param padding Padding type (0=VALID, 1=SAME)
     * @param weight Pointer to weight tensor
     * @param bias Pointer to bias tensor
     */
    Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
           uint32_t output_channel_size, int32_t kernel_row_size, uint32_t kernel_col_size,
           uint32_t stride_row, uint32_t stride_col, uint32_t padding, uint32_t groups,
           const float* weight, const float* bias);

    /**
     * @brief Forward pass for floating-point Conv2d
     * @param input Input tensor (float)
     * @param output Output tensor (float)
     */
    void forward(float* input, float* output);
};

#elif defined(DYNAMIC_QUANTIZATION_PER_TENSOR)

// ======================================================================
// Dynamic Quantization Conv2d (Per-Tensor)
// ======================================================================

/**
 * @brief Dynamically quantized 2D convolution layer
 * 
 * Uses int8_t weights with float input/output and per-tensor scaling
 */
class Conv2d : public Layer {
private:
    // Input tensor dimensions
    uint32_t input_channel_size;  ///< Number of input channels
    uint32_t input_row_size;      ///< Height of input feature map
    uint32_t input_col_size;      ///< Width of input feature map

    // Output tensor dimensions
    uint32_t output_channel_size; ///< Number of output channels
    uint32_t output_row_size;     ///< Height of output feature map
    uint32_t output_col_size;     ///< Width of output feature map

    // Kernel parameters
    uint32_t kernel_row_size;     ///< Height of convolution kernel
    uint32_t kernel_col_size;     ///< Width of convolution kernel

    // Operation parameters
    uint32_t stride_row;         ///< Vertical stride
    uint32_t stride_col;         ///< Horizontal stride
    uint32_t padding;            ///< Padding type (0=VALID, 1=SAME)
    uint32_t groups;

    // Quantization parameters
    const int8_t* weight;       ///< Pointer to quantized weight tensor
    float weight_scale;         ///< Scale factor for weights
    const float* bias;          ///< Pointer to bias tensor (float)

public:
    /**
     * @brief Constructor for dynamically quantized Conv2d
     * @param weight_scale Scale factor for quantized weights
     * @param other parameters same as floating-point version
     */
    Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
           uint32_t output_channel_size, int32_t kernel_row_size, uint32_t kernel_col_size,
           uint32_t stride_row, uint32_t stride_col, uint32_t padding, uint32_t groups,
           const int8_t* weight, float weight_scale, const float* bias);

    /**
     * @brief Forward pass for dynamically quantized Conv2d
     * @param input Input tensor (float)
     * @param output Output tensor (float)
     */
    void forward(float* input, float* output);
};

#elif defined(STATIC_QUANTIZATION_PER_TENSOR)

// ======================================================================
// Static Quantization Conv2d (Per-Tensor)
// ======================================================================

/**
 * @brief Statically quantized 2D convolution layer
 * 
 * Uses int8_t for both weights and activations with per-tensor scaling
 */
class Conv2d : public Layer {
private:
    // Input tensor dimensions
    uint32_t input_channel_size;  ///< Number of input channels
    uint32_t input_row_size;      ///< Height of input feature map
    uint32_t input_col_size;      ///< Width of input feature map

    // Output tensor dimensions
    uint32_t output_channel_size; ///< Number of output channels
    uint32_t output_row_size;     ///< Height of output feature map
    uint32_t output_col_size;     ///< Width of output feature map

    // Kernel parameters
    uint32_t kernel_row_size;     ///< Height of convolution kernel
    uint32_t kernel_col_size;     ///< Width of convolution kernel

    // Operation parameters
    uint32_t stride_row;         ///< Vertical stride
    uint32_t stride_col;         ///< Horizontal stride
    uint32_t padding;            ///< Padding type (0=VALID, 1=SAME)
    uint32_t groups;

    // Quantization parameters
    float output_scale;          ///< Output tensor scale factor
    int8_t output_zero_point;    ///< Output tensor zero point
    int8_t input_zero_point;     ///< Input tensor zero point

    // Weight and bias tensors
    const int8_t* weight;       ///< Pointer to quantized weight tensor
    const int32_t* bias;        ///< Pointer to quantized bias tensor
    float bias_scale;           ///< Bias scale factor

public:
    /**
     * @brief Constructor for statically quantized Conv2d
     * @param output_scale Output tensor scale factor
     * @param output_zero_point Output tensor zero point
     * @param input_zero_point Input tensor zero point
     * @param bias_scale Bias scale factor
     * @param other parameters same as other versions
     */
    Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
           uint32_t output_channel_size, int32_t kernel_row_size, uint32_t kernel_col_size,
           uint32_t stride_row, uint32_t stride_col, uint32_t padding, uint32_t groups,
           float output_scale, int8_t output_zero_point, int8_t input_zero_point,
           const int8_t* weight, const int32_t* bias, float bias_scale);

    /**
     * @brief Forward pass for statically quantized Conv2d
     * @param input Input tensor (int8_t)
     * @param output Output tensor (int8_t)
     */
    void forward(int8_t* input, int8_t* output);
};

#endif // QUANTIZATION_NONE

#endif // CONV_H
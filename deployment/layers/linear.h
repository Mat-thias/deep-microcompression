/**
 * @file linear.h
 * @brief Linear (fully-connected) layer interface with support for:
 *       1. Non-quantized models (float)
 *       2. Dynamic quantized models per tensor (float input + quantized weights)
 *       3. Static quantized models per tensor (all quantized)
 */

#ifndef LINEAR_H
#define LINEAR_H

#include "layer.h"

#if defined(QUANTIZATION_NONE)

/**
 * @class Linear
 * @brief Floating-point fully-connected layer
 */
class Linear : public Layer {
private:
    uint32_t input_size;      ///< Number of input features
    uint32_t output_size;     ///< Number of output neurons
    const float* weight;      ///< Weight matrix (row-major, shape [output_size, input_size])
    const float* bias;        ///< Bias vector (size: output_size)

public:
    /**
     * @brief Constructor for floating-point Linear layer
     * @param output_size Number of output neurons
     * @param input_size Number of input features
     * @param weight Pointer to weight matrix
     * @param bias Pointer to bias vector
     */
    Linear(uint32_t output_size, uint32_t input_size, 
          const float* weight, const float* bias);

    /**
     * @brief Forward pass for floating-point Linear layer
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    void forward(float* input, float* output) override;
};

#elif defined(DYNAMIC_QUANTIZATION_PER_TENSOR)

/**
 * @class Linear
 * @brief Dynamically quantized fully-connected layer (weights only)
 */
class Linear : public Layer {
private:
    uint32_t input_size;      ///< Number of input features
    uint32_t output_size;     ///< Number of output neurons
    const int8_t* weight;     ///< Quantized weight matrix
    float weight_scale;       ///< Scaling factor for weights
    const float* bias;        ///< Floating-point bias vector (size: output_size)

public:
    /**
     * @brief Constructor for dynamically quantized Linear layer
     * @param output_size Number of output neurons
     * @param input_size Number of input features
     * @param weight Pointer to quantized weight matrix
     * @param weight_scale Scaling factor for weights
     * @param bias Pointer to bias vector
     */
    Linear(uint32_t output_size, uint32_t input_size,
          const int8_t* weight, float weight_scale,
          const float* bias);

    /**
     * @brief Forward pass for dynamically quantized Linear layer
     * @param input Pointer to input tensor (float)
     * @param output Pointer to output tensor (float)
     */
    void forward(float* input, float* output) override;
};

#elif defined(STATIC_QUANTIZATION_PER_TENSOR)

/**
 * @class Linear
 * @brief Statically quantized fully-connected layer (all tensors quantized)
 */
class Linear : public Layer {
private:
    uint32_t input_size;      ///< Number of input features
    uint32_t output_size;     ///< Number of output neurons
    float output_scale;       ///< Output tensor scaling factor
    int8_t output_zero_point; ///< Output tensor zero point
    int8_t input_zero_point;  ///< Input tensor zero point
    const int8_t* weight;     ///< Quantized weight matrix
    const int32_t* bias;      ///< Quantized bias vector (size: output_size)
    float bias_scale;         ///< Bias scaling factor

public:
    /**
     * @brief Constructor for statically quantized Linear layer
     * @param output_size Number of output neurons
     * @param input_size Number of input features
     * @param output_scale Output tensor scaling factor
     * @param output_zero_point Output tensor zero point
     * @param input_zero_point Input tensor zero point
     * @param weight Pointer to quantized weight matrix
     * @param bias Pointer to quantized bias vector
     * @param bias_scale Bias scaling factor
     */
    Linear(uint32_t output_size, uint32_t input_size,
          float output_scale, int8_t output_zero_point,
          int8_t input_zero_point, const int8_t* weight,
          const int32_t* bias, float bias_scale);

    /**
     * @brief Forward pass for statically quantized Linear layer
     * @param input Pointer to quantized input tensor (int8_t)
     * @param output Pointer to quantized output tensor (int8_t)
     */
    void forward(int8_t* input, int8_t* output) override;
};

#endif // QUANTIZATION_NONE

#endif // LINEAR_H
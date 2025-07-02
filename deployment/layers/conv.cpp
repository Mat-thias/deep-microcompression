/**
 * @file conv.cpp
 * @brief Implementation of 2D convolution layer with support for:
 *      1. None quantized model
 *      2. Dynamic quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 *      3. Static quantized model per tensor
 *          - 8 bit
 *          - 4 bit
 * 
 * Supports 4-bit and 8-bit weight packing for quantized modes.
 */

#include "conv.h"

// Padding type constants
#define PADDING_VALID 0
#define PADDING_SAME  1

#if defined(QUANTIZATION_NONE) || (!defined(DYNAMIC_QUANTIZATION_PER_TENSOR) && !defined(DYNAMIC_QUANTIZATION_PER_CHANNEL) \
                               && !defined(STATIC_QUANTIZATION_PER_TENSOR) && !defined(STATIC_QUANTIZATION_PER_CHANNEL))

// ======================================================================
// Floating-Point Implementation
// ======================================================================

/**
 * @brief Constructor for floating-point Conv2d layer
 * @param input_channel_size Number of input channels
 * @param input_row_size Input height in pixels
 * @param input_col_size Input width in pixels
 * @param output_channel_size Number of output channels
 * @param kernel_row_size Kernel height
 * @param kernel_col_size Kernel width
 * @param stride_row Vertical stride
 * @param stride_col Horizontal stride
 * @param padding Padding type (PADDING_VALID or PADDING_SAME)
 * @param weight Pointer to weight tensor (float)
 * @param bias Pointer to bias tensor (float)
 */
Conv2d::Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
               uint32_t output_channel_size, int32_t kernel_row_size, uint32_t kernel_col_size,
               uint32_t stride_row, uint32_t stride_col, uint32_t padding, uint32_t groups,
               const float* weight, const float* bias) {
    
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->padding = padding;
    this->groups = groups;
    
    this->weight = weight;
    this->bias = bias;

    // Compute output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;
}

/**
 * @brief Forward pass for floating-point Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
void Conv2d::forward(float* input, float* output) {
    uint32_t output_index;

    uint32_t input_channel_per_group = this->input_channel_size / this->groups;
    uint32_t output_channel_per_group = this->output_channel_size / this->groups;

    uint32_t n, k;

    switch (this->padding) {
    case PADDING_VALID:
        for (uint32_t g = 0; g < this->groups; g++){
            // Output channel loop
            for (uint32_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                n = g * output_channel_per_group + c_out;
                // Output spatial dimensions loops
                for (uint32_t m = 0; m < this->output_row_size; m++) {
                    for (uint32_t l = 0; l < this->output_col_size; l++) {
                        
                        // Calculate output index
                        output_index = (n * this->output_row_size * this->output_col_size) + 
                                    (m * this->output_col_size) + 
                                    l;
                        output[output_index] = 0;

                        for (uint32_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                            k = g * input_channel_per_group + c_in;
                            for (uint32_t j = 0; j < kernel_row_size; j++) {
                                for (uint32_t i = 0; i < kernel_col_size; i++) {
                                    
                                    // Convolution operation
                                    output[output_index] += 
                                        input[(k * this->input_row_size * this->input_col_size) +
                                            ((j + m * this->stride_row) * this->input_col_size) + 
                                            (i + l * this->stride_col)] *
                                        this->weight[(n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                                    (j * this->kernel_col_size) + 
                                                    i];
                                }
                            }
                        }
                        // Add bias
                        output[output_index] += this->bias[n];
                    }
                }
            }
        }
        // // Output channel loop
        // for (uint32_t n = 0; n < this->output_channel_size; n++) {
        //     // Output spatial dimensions loops
        //     for (uint32_t m = 0; m < this->output_row_size; m++) {
        //         for (uint32_t l = 0; l < this->output_col_size; l++) {
                    
        //             // Calculate output index
        //             output_index = (n * this->output_row_size * this->output_col_size) + 
        //                           (m * this->output_col_size) + 
        //                           l;
        //             output[output_index] = 0;

        //             // Input channel and kernel loops
        //             for (uint32_t k = 0; k < input_channel_size; k++) {
        //                 for (uint32_t j = 0; j < kernel_row_size; j++) {
        //                     for (uint32_t i = 0; i < kernel_col_size; i++) {
                                
        //                         // Convolution operation
        //                         output[output_index] += 
        //                             input[(k * this->input_row_size * this->input_col_size) +
        //                                  ((j + m * this->stride_row) * this->input_col_size) + 
        //                                  (i + l * this->stride_col)] *
        //                             this->weight[(n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
        //                                         (k * this->kernel_row_size * kernel_col_size) + 
        //                                         (j * this->kernel_col_size) + 
        //                                         i];
        //                     }
        //                 }
        //             }
        //             // Add bias
        //             output[output_index] += this->bias[n];
        //         }
        //     }
        // }
        break;

    case PADDING_SAME:
        // TODO: Implement same padding
        break;
    }
}

#elif defined(DYNAMIC_QUANTIZATION_PER_TENSOR)

// ======================================================================
// Dynamic Quantization Implementation (Per-Tensor)
// ======================================================================

/**
 * @brief Constructor for dynamically quantized Conv2d layer
 * @param weight_scale Scale factor for quantized weights
 */
Conv2d::Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
               uint32_t output_channel_size, int32_t kernel_row_size, uint32_t kernel_col_size,
               uint32_t stride_row, uint32_t stride_col, uint32_t padding, uint32_t groups,
               const int8_t* weight, float weight_scale, const float* bias) {
    
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;
    
    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->padding = padding;
    this->groups = groups;
    
    this->weight = weight;
    this->weight_scale = weight_scale;
    this->bias = bias;

    // Compute output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;
}

/**
 * @brief Forward pass for dynamically quantized Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
void Conv2d::forward(float* input, float* output) {
    int output_index;

    uint32_t input_channel_per_group = this->input_channel_size / this->groups;
    uint32_t output_channel_per_group = this->output_channel_size / this->groups;

    uint32_t n, k;

    switch (this->padding) {
    case PADDING_VALID:
        for (uint32_t g = 0; g < this->groups; g++){
            // Output channel loop
            for (uint32_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                n = g * output_channel_per_group + c_out;
                // Output spatial dimensions loops
                for (uint32_t m = 0; m < this->output_row_size; m++) {
                    for (uint32_t l = 0; l < this->output_col_size; l++) {
                        
                        // Calculate output index
                        output_index = (n * this->output_row_size * this->output_col_size) + 
                                    (m * this->output_col_size) + 
                                    l;
                        output[output_index] = 0;

                        for (uint32_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                            k = g * input_channel_per_group + c_in;
                            for (uint32_t j = 0; j < kernel_row_size; j++) {
                                for (uint32_t i = 0; i < kernel_col_size; i++) {
#if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
                                    
                                    // 8-bit quantization path
                                    output[output_index] += 
                                        input[(k * this->input_row_size * this->input_col_size) +
                                            ((j + m * this->stride_row) * this->input_col_size) + 
                                            (i + l * this->stride_col)] *
                                        this->weight[(n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                    (c_in * this->kernel_row_size * kernel_col_size) + 
                                                    (j * this->kernel_col_size) + 
                                                    i] * this->weight_scale;
#elif QUANTIZATION_BITWIDTH == 4
                                // 4-bit quantization path
                                output[output_index] += 
                                    input[(k * this->input_row_size * this->input_col_size) +
                                         ((j + m * this->stride_row) * this->input_col_size) + 
                                         (i + l * this->stride_col)] *
                                    (((int8_t)(((this->weight[((n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
                                         (k * this->kernel_row_size * kernel_col_size) + 
                                         (j * this->kernel_col_size) + 
                                         i) >> 1] >> (
                                         (((n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                         (k * c_in * kernel_col_size) + 
                                         (j * this->kernel_col_size) + 
                                         i) & 1) << 2)) & 0x0F) << 4)) >> 4) * this->weight_scale;
#endif // QUANTIZATION_BITWIDTH
                                }
                            }
                        }
                        // Add bias
                        output[output_index] += this->bias[n];
                    }
                }
            }
        }

//         // Output channel loop
//         for (uint32_t n = 0; n < this->output_channel_size; n++) {
//             // Output spatial dimensions loops
//             for (uint32_t m = 0; m < this->output_row_size; m++) {
//                 for (uint32_t l = 0; l < this->output_col_size; l++) {
                    
//                     // Calculate output index
//                     output_index = (n * this->output_row_size * this->output_col_size) + 
//                                   (m * this->output_col_size) + 
//                                   l;
//                     output[output_index] = 0;

//                     // Input channel and kernel loops
//                     for (uint32_t k = 0; k < input_channel_size; k++) {
//                         for (uint32_t j = 0; j < kernel_row_size; j++) {
//                             for (uint32_t i = 0; i < kernel_col_size; i++) {
                                
// #if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
//                                 // 8-bit quantization path
//                                 output[output_index] += 
//                                     input[(k * this->input_row_size * this->input_col_size) +
//                                          ((j + m * this->stride_row) * this->input_col_size) + 
//                                          (i + l * this->stride_col)] *
//                                     this->weight[(n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
//                                                 (k * this->kernel_row_size * kernel_col_size) + 
//                                                 (j * this->kernel_col_size) + 
//                                                 i] * this->weight_scale;

// #elif QUANTIZATION_BITWIDTH == 4
//                                 // 4-bit quantization path
//                                 output[output_index] += 
//                                     input[(k * this->input_row_size * this->input_col_size) +
//                                          ((j + m * this->stride_row) * this->input_col_size) + 
//                                          (i + l * this->stride_col)] *
//                                     (((int8_t)(((this->weight[((n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
//                                          (k * this->kernel_row_size * kernel_col_size) + 
//                                          (j * this->kernel_col_size) + 
//                                          i) >> 1] >> (
//                                          (((n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
//                                          (k * this->kernel_row_size * kernel_col_size) + 
//                                          (j * this->kernel_col_size) + 
//                                          i) & 1) << 2)) & 0x0F) << 4)) >> 4) * this->weight_scale;
// #endif // QUANTIZATION_BITWIDTH
//                             }
//                         }
//                     }
//                     // Add bias
//                     output[output_index] += this->bias[n];
//                 }
//             }
//         }
        break;

    case PADDING_SAME:
        // TODO: Implement same padding
        break;
    }
}

#elif defined(STATIC_QUANTIZATION_PER_TENSOR)

// ======================================================================
// Static Quantization Implementation (Per-Tensor)
// ======================================================================

/**
 * @brief Constructor for statically quantized Conv2d layer
 * @param output_scale Output tensor scale factor
 * @param output_zero_point Output tensor zero point
 * @param input_zero_point Input tensor zero point
 * @param bias_scale Bias scale factor
 */
Conv2d::Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
               uint32_t output_channel_size, int32_t kernel_row_size, uint32_t kernel_col_size,
               uint32_t stride_row, uint32_t stride_col, uint32_t padding, uint32_t groups,
               float output_scale, int8_t output_zero_point, int8_t input_zero_point,
               const int8_t* weight, const int32_t* bias, float bias_scale) {

    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->output_channel_size = output_channel_size;
    this->kernel_row_size = kernel_row_size;
    this->kernel_col_size = kernel_col_size;

    this->stride_row = stride_row;
    this->stride_col = stride_col;
    this->padding = padding;
    this->groups = groups;

    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->input_zero_point = input_zero_point;

    this->weight = weight;
    this->bias = bias;
    this->bias_scale = bias_scale;

    // Compute output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;
}

/**
 * @brief Forward pass for statically quantized Conv2d
 * @param input Input tensor (int8_t)
 * @param output Output tensor (int8_t)
 */
void Conv2d::forward(int8_t* input, int8_t* output) {
    int output_index;

    uint32_t input_channel_per_group = this->input_channel_size / this->groups;
    uint32_t output_channel_per_group = this->output_channel_size / this->groups;

    int32_t output_temp;
    uint32_t n, k;

    switch (this->padding) {
    case PADDING_VALID:

        for (uint32_t g = 0; g < this->groups; g++){
            // Output channel loop
            for (uint32_t c_out = 0; c_out < output_channel_per_group; c_out++) {
                n = g * output_channel_per_group + c_out;
                // Output spatial dimensions loops
                for (uint32_t m = 0; m < this->output_row_size; m++) {
                    for (uint32_t l = 0; l < this->output_col_size; l++) {
                        
                        // Calculate output index
                        output_index = (n * this->output_row_size * this->output_col_size) + 
                                    (m * this->output_col_size) + 
                                    l;
                        output_temp = 0;

                        for (uint32_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                            k = g * input_channel_per_group + c_in;
                            for (uint32_t j = 0; j < kernel_row_size; j++) {
                                for (uint32_t i = 0; i < kernel_col_size; i++) {
#if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
                                // 8-bit quantization path
                                output_temp += 
                                    ((int32_t)input[(k * this->input_row_size * this->input_col_size) +
                                                   ((j + m * this->stride_row) * this->input_col_size) + 
                                                   (i + l * this->stride_col)] - this->input_zero_point) *
                                    (int32_t)this->weight[(n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                         (c_in * this->kernel_row_size * kernel_col_size) + 
                                                         (j * this->kernel_col_size) + 
                                                         i];
#elif QUANTIZATION_BITWIDTH == 4
                                // 4-bit quantization path
                                output_temp += 
                                    ((int32_t)input[(k * this->input_row_size * this->input_col_size) +
                                                   ((j + m * this->stride_row) * this->input_col_size) + 
                                                   (i + l * this->stride_col)] - this->input_zero_point) *
                                    (int32_t)(((int8_t)(((this->weight[((n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                   (c_in * this->kernel_row_size * kernel_col_size) + 
                                                   (j * this->kernel_col_size) + 
                                                   i) >> 1] >> (
                                                   (((n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                   (c_in * this->kernel_row_size * kernel_col_size) + 
                                                   (j * this->kernel_col_size) + 
                                                   i) & 1) << 2)) & 0x0F) << 4)) >> 4);
#endif // QUANTIZATION_BITWIDTH
                                }
                            }
                        }
                    // Apply bias, scaling and clamping
                    output_temp += this->bias[n];
                    output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
                    output_temp += this->output_zero_point;
                    
                    // Clamp to int8_t range
                    output[output_index] = output_temp < -128 ? (int8_t)-128 : 
                                         (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);
                    }
                }
            }
        }
        
//         // Output channel loop
//         for (uint32_t n = 0; n < this->output_channel_size; n++) {
//             // Output spatial dimensions loops
//             for (uint32_t m = 0; m < this->output_row_size; m++) {
//                 for (uint32_t l = 0; l < this->output_col_size; l++) {
                    
//                     // Calculate output index
//                     output_index = (n * this->output_row_size * this->output_col_size) + 
//                                   (m * this->output_col_size) + 
//                                   l;
//                     output_temp = 0;

//                     // Input channel and kernel loops
//                     for (uint32_t k = 0; k < input_channel_size; k++) {
//                         for (uint32_t j = 0; j < kernel_row_size; j++) {
//                             for (uint32_t i = 0; i < kernel_col_size; i++) {
                                
// #if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
//                                 // 8-bit quantization path
//                                 output_temp += 
//                                     ((int32_t)input[(k * this->input_row_size * this->input_col_size) +
//                                                    ((j + m * this->stride_row) * this->input_col_size) + 
//                                                    (i + l * this->stride_col)] - this->input_zero_point) *
//                                     (int32_t)this->weight[(n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
//                                                          (k * this->kernel_row_size * kernel_col_size) + 
//                                                          (j * this->kernel_col_size) + 
//                                                          i];

// #elif QUANTIZATION_BITWIDTH == 4
//                                 // 4-bit quantization path
//                                 output_temp += 
//                                     ((int32_t)input[(k * this->input_row_size * this->input_col_size) +
//                                                    ((j + m * this->stride_row) * this->input_col_size) + 
//                                                    (i + l * this->stride_col)] - this->input_zero_point) *
//                                     (int32_t)(((int8_t)(((this->weight[((n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
//                                                    (k * this->kernel_row_size * kernel_col_size) + 
//                                                    (j * this->kernel_col_size) + 
//                                                    i) >> 1] >> (
//                                                    (((n * this->input_channel_size * this->kernel_row_size * this->kernel_col_size) +
//                                                    (k * this->kernel_row_size * kernel_col_size) + 
//                                                    (j * this->kernel_col_size) + 
//                                                    i) & 1) << 2)) & 0x0F) << 4)) >> 4);
// #endif // QUANTIZATION_BITWIDTH
//                             }
//                         }
//                     }
                    
//                     // Apply bias, scaling and clamping
//                     output_temp += this->bias[n];
//                     output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
//                     output_temp += this->output_zero_point;
                    
//                     // Clamp to int8_t range
//                     output[output_index] = output_temp < -128 ? (int8_t)-128 : 
//                                          (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);
//                 }
//             }
//         }
        break;

    case PADDING_SAME:
        // TODO: Implement same padding
        break;
    }
}

#endif // QUANTIZATION_NONE
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


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME == NONE


Conv2d::Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
               uint32_t output_channel_size, uint32_t kernel_row_size, uint32_t kernel_col_size,
               uint32_t stride_row, uint32_t stride_col, Padding_t padding, uint32_t groups,
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
    this->output_row_size = ((this->input_row_size + this->padding.padding_top + this->padding.padding_bottom - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + this->padding.padding_left + this->padding.padding_right - this->kernel_col_size) / this->stride_col) + 1;
}

/**
 * @brief Forward pass for floating-point Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
void Conv2d::forward(float* input, float* output) {
    float output_temp;

    uint32_t input_channel_per_group = this->input_channel_size / this->groups;
    uint32_t output_channel_per_group = this->output_channel_size / this->groups;

    uint32_t n, k;

    uint32_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint32_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint32_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint32_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint32_t m = 0; m < this->output_row_size; m++) {
                for (uint32_t l = 0; l < this->output_col_size; l++) {
                    
                    if (this->bias) {
                        output_temp = this->bias[n];
                    }
                    else {
                        output_temp = 0;
                    }

                    for (uint32_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint32_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint32_t i = 0; i < this->kernel_col_size; i++) {

                                // Convolution operation
                                output_temp += 
                                    input[(k * padded_row_size * padded_col_size) +
                                        ((j + m * this->stride_row) * padded_col_size) + 
                                        (i + l * this->stride_col)] *
                                    this->weight[(n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                (c_in * this->kernel_row_size * kernel_col_size) + 
                                                (j * this->kernel_col_size) + 
                                                i];
                            }
                        }
                    }
                    output[(n * this->output_row_size * this->output_col_size) + 
                            (m * this->output_col_size) + 
                            l] = output_temp;
                }
            }
        }
    }
}




#elif QUANTIZATION_SCHEME == DYNAMIC // QUANTIZATION_SCHEME

/**
 * @brief Constructor for dynamically quantized Conv2d layer
 * @param weight_scale Scale factor for quantized weights
 */
Conv2d::Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
               uint32_t output_channel_size, uint32_t kernel_row_size, uint32_t kernel_col_size,
               uint32_t stride_row, uint32_t stride_col, Padding_t padding, uint32_t groups,
               const int8_t* weight, const float* bias, float weight_scale) {

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
    this->weight_scale = weight_scale;

    // Compute output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_col_size) / this->stride_col) + 1;


    // Compute output dimensions
    this->output_row_size = ((this->input_row_size + this->padding.padding_top + this->padding.padding_bottom - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + this->padding.padding_left + this->padding.padding_right - this->kernel_col_size) / this->stride_col) + 1;
}


/**
 * @brief Forward pass for floating-point Conv2d
 * @param input Input tensor (float)
 * @param output Output tensor (float)
 */
void Conv2d::forward(float* input, float* output) {
    // uint32_t output_index;
    float output_temp;

    uint32_t input_channel_per_group = this->input_channel_size / this->groups;
    uint32_t output_channel_per_group = this->output_channel_size / this->groups;

    uint32_t n, k;

    uint32_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint32_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

    for (uint32_t g = 0; g < this->groups; g++){
        // Output channel loop
        for (uint32_t c_out = 0; c_out < output_channel_per_group; c_out++) {
            n = g * output_channel_per_group + c_out;
            // Output spatial dimensions loops
            for (uint32_t m = 0; m < this->output_row_size; m++) {
                for (uint32_t l = 0; l < this->output_col_size; l++) {
                    
                    output_temp = 0;
                    for (uint32_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint32_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint32_t i = 0; i < this->kernel_col_size; i++) {

                                // Convolution operation
                                output_temp += 
                                    input[(k * padded_row_size * padded_col_size) +
                                        ((j + m * this->stride_row) * padded_col_size) + 
                                        (i + l * this->stride_col)] *
                                    this->weight[(n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                (c_in * this->kernel_row_size * kernel_col_size) + 
                                                (j * this->kernel_col_size) + 
                                                i];
                            }
                        }
                    }
                    output[(n * this->output_row_size * this->output_col_size) + 
                        (m * this->output_col_size) + 
                        l] = this->bias ? 
                        output_temp * this->weight_scale + this->bias[n]:
                        output_temp * this->weight_scale;    
                }
            }
        }
    }
}


#elif QUANTIZATION_SCHEME == STATIC // QUANTIZATION_SCHEME

Conv2d::Conv2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
               uint32_t output_channel_size, uint32_t kernel_row_size, uint32_t kernel_col_size,
               uint32_t stride_row, uint32_t stride_col, Padding_t padding, uint32_t groups,
               const int8_t* weight, const int32_t* bias, float output_scale, 
               int8_t output_zero_point, int8_t input_zero_point,  float bias_scale) {
                
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

    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->input_zero_point = input_zero_point;

    this->bias_scale = bias_scale;

    this->output_row_size = ((this->input_row_size + this->padding.padding_top + this->padding.padding_bottom - this->kernel_row_size) / this->stride_row) + 1;
    this->output_col_size = ((this->input_col_size + this->padding.padding_left + this->padding.padding_right - this->kernel_col_size) / this->stride_col) + 1;

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

    uint32_t padded_row_size = this->input_row_size + this->padding.padding_top + this->padding.padding_bottom;
    uint32_t padded_col_size = this->input_col_size + this->padding.padding_left + this->padding.padding_right;

    pad_input(input, this->input_zero_point, this->padding, input_channel_size, input_row_size, input_col_size, padded_row_size, padded_col_size);

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
                    if (this->bias) {
                        output_temp = this->bias[n];
                    }
                    else {
                        output_temp = 0;
                    }

                    for (uint32_t c_in = 0; c_in < input_channel_per_group; c_in++) {
                        k = g * input_channel_per_group + c_in;
                        for (uint32_t j = 0; j < this->kernel_row_size; j++) {
                            for (uint32_t i = 0; i < this->kernel_col_size; i++) {
                                int32_t in = ((int32_t)input[(k * this->input_row_size * this->input_col_size) +
                                                   ((j + m * this->stride_row) * this->input_col_size) + 
                                                   (i + l * this->stride_col)] - this->input_zero_point);
                                int32_t we = (int32_t)this->weight[(n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                         (c_in * this->kernel_row_size * kernel_col_size) + 
                                                         (j * this->kernel_col_size) + 
                                                         i];

                                int32_t te = ((int32_t)input[(k * this->input_row_size * this->input_col_size) +
                                                   ((j + m * this->stride_row) * this->input_col_size) + 
                                                   (i + l * this->stride_col)] - this->input_zero_point) *
                                    (int32_t)this->weight[(n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                         (c_in * this->kernel_row_size * kernel_col_size) + 
                                                         (j * this->kernel_col_size) + 
                                                         i];




                                // Convolution operation
                                output_temp += 
                                    ((int32_t)input[(k * padded_row_size * padded_col_size) +
                                                   ((j + m * this->stride_row) * padded_col_size) + 
                                                   (i + l * this->stride_col)] - this->input_zero_point) *
                                    (int32_t)this->weight[(n * input_channel_per_group * this->kernel_row_size * this->kernel_col_size) +
                                                         (c_in * this->kernel_row_size * kernel_col_size) + 
                                                         (j * this->kernel_col_size) + 
                                                         i];
                            }
                        }
                    }
                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
                    output_temp += this->output_zero_point;
                    
                    // Clamp to int8_t range
                    output[output_index] = output_temp < -128 ? (int8_t)-128 : 
                                         (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);

                }
            }
        }
    }
}


#endif // QUANTIZATION_SCHEME

/**
 * @file pooling.cpp
 * @brief Implementation of MaxPool2d layer with support for:
 *       1. Non-quantized models (float)
 *       2. Static quantized models per tensor (int8_t)
 */

#include "pooling.h"

#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC



/**
 * @brief Constructor for floating-point MaxPool2d layer
 * @param input_channel_size Number of input channels
 * @param input_row_size Height of input feature map
 * @param input_col_size Width of input feature map
 * @param kernel_size Size of pooling window (square)
 * @param stride Stride of pooling operation
 * @param Padding Padding size (currently unused, reserved for future)
 */
MaxPool2d::MaxPool2d(uint32_t input_channel_size, uint32_t input_row_size, 
                    uint32_t input_col_size, uint32_t kernel_size, 
                    uint32_t stride, uint32_t padding) {
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // Calculate output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_size) / this->stride) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_size) / this->stride) + 1;
}

/**
 * @brief Forward pass for floating-point MaxPool2d
 * @param input Pointer to input tensor (float) in CHW format
 * @param output Pointer to output tensor (float) in CHW format
 * 
 * Performs 2D max pooling operation with the specified kernel size and stride.
 * The input is assumed to be in CHW (channels, height, width) format.
 */
void MaxPool2d::forward(float* input, float* output) {
    float temp, input_val;

    // Loop over all channels
    for (uint32_t n = 0; n < this->input_channel_size; n++) {
        // Loop over output spatial dimensions
        for (uint32_t m = 0; m < this->output_row_size; m++) {
            for (uint32_t l = 0; l < this->output_col_size; l++) {
                // Initialize max value to smallest possible float
                temp = -FLT_MAX;

                // Iterate through pooling window
                for (uint32_t j = 0; j < this->kernel_size; j++) {
                    for (uint32_t i = 0; i < this->kernel_size; i++) {
                        // Calculate input index
                        uint32_t input_idx = (n * this->input_row_size * this->input_col_size) +
                                          ((m * this->stride + j) * this->input_col_size) +
                                          (l * this->stride + i);
                        input_val = input[input_idx];

                        // Update max value
                        if (input_val > temp) {
                            temp = input_val;
                        }
                    }
                }

                // Store max value in output
                // uint32_t output_idx = (n * this->output_row_size * this->output_col_size) +
                //                      (m * this->output_col_size) + l;
                output[(n * this->output_row_size * this->output_col_size) +
                        (m * this->output_col_size) + l] = temp;
            }
        }
    }
}



AvgPool2d::AvgPool2d(uint32_t input_channel_size, uint32_t input_row_size, 
                    uint32_t input_col_size, uint32_t kernel_size, 
                    uint32_t stride, uint32_t padding) {
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // Calculate output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_size) / this->stride) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_size) / this->stride) + 1;
}

void AvgPool2d::forward(float* input, float* output) {
    float total;
    uint8_t pool_size = this->kernel_size * this->kernel_size;


    // Loop over all channels
    for (uint32_t n = 0; n < this->input_channel_size; n++) {
        // Loop over output spatial dimensions
        for (uint32_t m = 0; m < this->output_row_size; m++) {
            for (uint32_t l = 0; l < this->output_col_size; l++) {
                // Initialize max value to smallest possible float
                total = 0;

                // Iterate through pooling window
                for (uint32_t j = 0; j < this->kernel_size; j++) {
                    for (uint32_t i = 0; i < this->kernel_size; i++) {
                        // Calculate input index
                        uint32_t input_idx = (n * this->input_row_size * this->input_col_size) +
                                          ((m * this->stride + j) * this->input_col_size) +
                                          (l * this->stride + i);
                        total += input[input_idx];
                    }
                }

                // uint32_t output_idx = (n * this->output_row_size * this->output_col_size) +
                //                      (m * this->output_col_size) + l;
                output[(n * this->output_row_size * this->output_col_size) +
                                     (m * this->output_col_size) + l] = total / pool_size;
            }
        }
    }
}


#else // QUANTIZATION_SCHEME

MaxPool2d::MaxPool2d(uint32_t input_channel_size, uint32_t input_row_size,
                    uint32_t input_col_size, uint32_t kernel_size,
                    uint32_t stride, uint32_t padding) {
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // Calculate output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_size) / this->stride) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_size) / this->stride) + 1;
}

void MaxPool2d::forward(int8_t* input, int8_t* output) {
    int8_t temp, input_val;

    // Loop over all channels
    for (uint32_t n = 0; n < this->input_channel_size; n++) {
        // Loop over output spatial dimensions
        for (uint32_t m = 0; m < this->output_row_size; m++) {
            for (uint32_t l = 0; l < this->output_col_size; l++) {
                // Initialize max value to smallest possible int8
                temp = -128;

                // Iterate through pooling window
                for (uint32_t j = 0; j < this->kernel_size; j++) {
                    for (uint32_t i = 0; i < this->kernel_size; i++) {
                        // Calculate input index
                        uint32_t input_idx = (n * this->input_row_size * this->input_col_size) +
                                          ((m * this->stride + j) * this->input_col_size) +
                                          (l * this->stride + i);
                        input_val = input[input_idx];

                        // Update max value
                        if (input_val > temp) {
                            temp = input_val;
                        }
                    }
                }

                // Store max value in output
                uint32_t output_idx = (n * this->output_row_size * this->output_col_size) +
                                     (m * this->output_col_size) + l;
                output[output_idx] = temp;
            }
        }
    }
}



AvgPool2d::AvgPool2d(uint32_t input_channel_size, uint32_t input_row_size,
                    uint32_t input_col_size, uint32_t kernel_size,
                    uint32_t stride, uint32_t padding) {
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;

    // Calculate output dimensions
    this->output_row_size = ((this->input_row_size - this->kernel_size) / this->stride) + 1;
    this->output_col_size = ((this->input_col_size - this->kernel_size) / this->stride) + 1;
}

void AvgPool2d::forward(int8_t* input, int8_t* output) {
    int16_t total;

    uint8_t pool_size = this->kernel_size * this->kernel_size;

    // Loop over all channels
    for (uint32_t n = 0; n < this->input_channel_size; n++) {
        // Loop over output spatial dimensions
        for (uint32_t m = 0; m < this->output_row_size; m++) {
            for (uint32_t l = 0; l < this->output_col_size; l++) {
                // Initialize max value to smallest possible int8
                total = 0;

                // Iterate through pooling window
                for (uint32_t j = 0; j < this->kernel_size; j++) {
                    for (uint32_t i = 0; i < this->kernel_size; i++) {
                        // Calculate input index
                        uint32_t input_idx = (n * this->input_row_size * this->input_col_size) +
                                          ((m * this->stride + j) * this->input_col_size) +
                                          (l * this->stride + i);
                        total += input[input_idx];

                    }
                }

                // Store max value in output
                uint32_t output_idx = (n * this->output_row_size * this->output_col_size) +
                                     (m * this->output_col_size) + l;
                output[output_idx] = (float)total / pool_size;
              
            }
        }
    }
}


#endif // QUANTIZATION_SCHEME

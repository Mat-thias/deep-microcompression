#include "batchnorm.h"


BatchNorm2d::BatchNorm2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
                const float* weight, const float* bias, const float* running_mean, const float* running_var) {
    
    
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    
    this->weight = weight;
    this->bias = bias;
    this->running_mean = running_mean;
    this->running_var = running_var;
}

void BatchNorm2d::forward(float* input, float* output) {

    for (uint32_t n = 0; n < this->input_channel_size; n++) {
        for (uint32_t m = 0; m < this->input_row_size; m++) {
            for (uint32_t l = 0; l < this->input_col_size; l++) {
                output[((n * this->input_row_size * this->input_col_size) + 
                        (m * this->input_col_size) + 
                        l)] = 
                        (
                            ((input[((n * this->input_row_size * this->input_col_size) + 
                            (m * this->input_col_size) + 
                            l)] - this->running_mean[n]) / this->running_var[n] * this->weight[n])
                            + this->bias[n]
                        );
            }
        }
    }
                        
}
#include "batchnorm.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME != STATIC

BatchNorm2d::BatchNorm2d(uint32_t input_channel_size, uint32_t input_row_size, uint32_t input_col_size,
                const float* folded_weight, const float* folded_bias) {
    
    
    // Store layer parameters
    this->input_channel_size = input_channel_size;
    this->input_row_size = input_row_size;
    this->input_col_size = input_col_size;
    
    this->folded_weight = folded_weight;
    this->folded_bias = folded_bias;
}

void BatchNorm2d::forward(float* input, float* output) {

    for (uint32_t n = 0; n < this->input_channel_size; n++) {
        for (uint32_t m = 0; m < this->input_row_size; m++) {
            for (uint32_t l = 0; l < this->input_col_size; l++) {

                // set_value(
                //     output, 
                //     ((n * this->input_row_size * this->input_col_size) + 
                //     (m * this->input_col_size) + l),
                //     (get_value(
                //         input, 
                //         ((n * this->input_row_size * this->input_col_size) + 
                //         (m * this->input_col_size) + l) 
                //     )  * this->folded_weight[n]) + this->folded_bias[n]
                // );

                output[((n * this->input_row_size * this->input_col_size) + 
                        (m * this->input_col_size) + 
                        l)] = 
                        (
                            (input[((n * this->input_row_size * this->input_col_size) + 
                            (m * this->input_col_size) + 
                            l)] * this->folded_weight[n])
                            + this->folded_bias[n]
                        );
            }
        }
    }
                        
}


#else // QUANTIZATION_SCHEME

#endif // QUANTIZATION_SCHEME

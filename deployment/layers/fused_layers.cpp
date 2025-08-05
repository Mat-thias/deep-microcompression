#include "fused_layers.h"


#if !defined(QUANTIZATION_SCHEME) || QUANTIZATION_SCHEME == NONE

void LinearReLU::forward(float* input, float* output) {
    float output_temp;
    for (uint32_t j = 0; j < this->output_size; j++) {
        if (this->bias) {output_temp =  this->bias[j];}
        else {output_temp =  0;}
        // Matrix-vector multiplication
        for (uint32_t i = 0; i < this->input_size; i++) {
            output_temp += input[i] * this->weight[(j * this->input_size) + i];
        }
        output[j] = relu(output_temp);
    }
}



void Conv2dReLU::forward(float* input, float* output) {
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
                            l] = relu(output_temp);
                }
            }
        }
    }
}


void LinearReLU6::forward(float* input, float* output) {
    float output_temp;
    for (uint32_t j = 0; j < this->output_size; j++) {
        if (this->bias) {output_temp =  this->bias[j];}
        else {output_temp =  0;}
        // Matrix-vector multiplication
        for (uint32_t i = 0; i < this->input_size; i++) {
            output_temp += input[i] * this->weight[(j * this->input_size) + i];
        }
        output[j] = relux(output_temp, 6.);
    }
}



void Conv2dReLU6::forward(float* input, float* output) {
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
                            l] = relux(output_temp, 6.);
                }
            }
        }
    }
}


#elif QUANTIZATION_SCHEME == DYNAMIC 


void LinearReLU::forward(float* input, float* output) {
    float output_temp;
    for (uint32_t j = 0; j < this->output_size; j++) {
        output_temp = 0;
        // Matrix-vector multiplication
        for (uint32_t i = 0; i < this->input_size; i++) {
            output_temp += input[i] * this->weight[(j * this->input_size) + i];
        }
        output_temp = this->bias ? 
        output_temp * this->weight_scale + this->bias[j] :
        output_temp * this->weight_scale;
        output[j] = relu(output_temp);
    }
}



void Conv2dReLU::forward(float* input, float* output) {
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
                    output_temp = this->bias ? 
                        output_temp * this->weight_scale + this->bias[n]:
                        output_temp * this->weight_scale;  

                    output[(n * this->output_row_size * this->output_col_size) + 
                            (m * this->output_col_size) + 
                            l] = relu(output_temp);
                }
            }
        }
    }
}


void LinearReLU6::forward(float* input, float* output) {
    float output_temp;
    for (uint32_t j = 0; j < this->output_size; j++) {
        output_temp = 0;
        // Matrix-vector multiplication
        for (uint32_t i = 0; i < this->input_size; i++) {
            output_temp += input[i] * this->weight[(j * this->input_size) + i];
        }
        output_temp = this->bias ? 
        output_temp * this->weight_scale + this->bias[j] :
        output_temp * this->weight_scale;
        output[j] = relux(output_temp, 6.);
    }
}



void Conv2dReLU6::forward(float* input, float* output) {
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
                    output_temp = this->bias ? 
                        output_temp * this->weight_scale + this->bias[n]:
                        output_temp * this->weight_scale;  

                    output[(n * this->output_row_size * this->output_col_size) + 
                            (m * this->output_col_size) + 
                            l] = relux(output_temp, 6.);

                }
            }
        }
    }
}

#elif QUANTIZATION_SCHEME == STATIC



void LinearReLU::forward(int8_t* input, int8_t* output) {
    int32_t output_temp;

    for (uint32_t j = 0; j < this->output_size; j++) {

        if (this->bias) {
            output_temp =  this->bias[j];
        }
        else {
            output_temp =  0;
        }
        for (uint32_t i = 0; i < this->input_size; i++) {
            #if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
                // 8-bit weights
                output_temp += ((int32_t)input[i] - this->input_zero_point) * 
                             (int32_t)this->weight[(j * this->input_size) + i];
            #elif QUANTIZATION_BITWIDTH == 4
                // 4-bit weights (packed 2 values per byte)
                output_temp += ((int32_t)input[i] - this->input_zero_point) * 
                    (int32_t)((int8_t)(((this->weight[((j * this->input_size) + i)>>1] >> 
                    ((((j * this->input_size) + i) & 1) << 2)) & 0x0F) << 4) >> 4);
            #endif // QUANTIZATION_BITWIDTH
        }

        output_temp = relu(output_temp);
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
        output_temp += this->output_zero_point;
        
        // Clamp to int8_t range
        output[j] = output_temp < -128 ? (int8_t)-128 : 
                   (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);
    }
}

void Conv2dReLU::forward(int8_t* input, int8_t* output) {
    int output_index;
    int32_t output_temp;

    uint32_t input_channel_per_group = this->input_channel_size / this->groups;
    uint32_t output_channel_per_group = this->output_channel_size / this->groups;

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

                    output_temp = relu(output_temp);

                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
                    output_temp += this->output_zero_point;
                    
                    // Clamp to int8_t range
                    output[(n * this->output_row_size * this->output_col_size) + 
                    (m * this->output_col_size) + 
                    l] = output_temp < -128 ? (int8_t)-128 : 
                                         (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);

                }
            }
        }
    }
}





void LinearReLU6::forward(int8_t* input, int8_t* output) {
    int32_t output_temp;
    int32_t six_point = (int32_t)((float)6. / this->bias_scale);

    for (uint32_t j = 0; j < this->output_size; j++) {

        if (this->bias) {
            output_temp =  this->bias[j];
        }
        else {
            output_temp =  0;
        }
        for (uint32_t i = 0; i < this->input_size; i++) {
            #if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
                // 8-bit weights
                output_temp += ((int32_t)input[i] - this->input_zero_point) * 
                             (int32_t)this->weight[(j * this->input_size) + i];
            #elif QUANTIZATION_BITWIDTH == 4
                // 4-bit weights (packed 2 values per byte)
                output_temp += ((int32_t)input[i] - this->input_zero_point) * 
                    (int32_t)((int8_t)(((this->weight[((j * this->input_size) + i)>>1] >> 
                    ((((j * this->input_size) + i) & 1) << 2)) & 0x0F) << 4) >> 4);
            #endif // QUANTIZATION_BITWIDTH
        }

        output_temp = relux(output_temp, six_point);
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
        output_temp += this->output_zero_point;
        
        // Clamp to int8_t range
        output[j] = output_temp < -128 ? (int8_t)-128 : 
                   (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);
    }
}

void Conv2dReLU6::forward(int8_t* input, int8_t* output) {
    int output_index;
    int32_t output_temp;
    int32_t six_point = (int32_t)((float)6. / this->bias_scale);

    uint32_t input_channel_per_group = this->input_channel_size / this->groups;
    uint32_t output_channel_per_group = this->output_channel_size / this->groups;

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
                            for (uint32_t i = 0; i < this->kernel_col_size; i++) {                                // Convolution operation
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

                    output_temp = relux(output_temp, six_point);

                    // Apply bias, scaling and clamping
                    output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
                    output_temp += this->output_zero_point;
                    
                    // Clamp to int8_t range
                    output[(n * this->output_row_size * this->output_col_size) + 
                    (m * this->output_col_size) + 
                    l] = output_temp < -128 ? (int8_t)-128 : 
                                         (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);

                }
            }
        }
    }
}



#endif // QUANTIZATION_SCHEME


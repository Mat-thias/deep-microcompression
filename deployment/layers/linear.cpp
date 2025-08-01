/**
 * @file linear.cpp
 * @brief Implementation of Linear (fully-connected) layer with support for:
 *       1. Non-quantized models (float)
 *       2. Dynamic quantized models per tensor (float input + quantized weights)
 *       3. Static quantized models per tensor (all quantized)
 */

#include "linear.h"



#ifdef DYNAMIC_QUANTIZATION_PER_TENSOR // QUANTIZATION_TYPE

/**
 * @brief Constructor for dynamically quantized Linear layer
 * @param output_size Number of output neurons
 * @param input_size Number of input features
 * @param weight Pointer to quantized weight matrix (int8_t)
 * @param weight_scale Scaling factor for weights
 * @param bias Pointer to floating-point bias vector
 */
Linear::Linear(uint32_t output_size, uint32_t input_size,
              const int8_t* weight, const float* bias,
              float weight_scale) {
    this->output_size = output_size;
    this->input_size = input_size;
    this->weight = weight;
    this->bias = bias;
    this->weight_scale = weight_scale;
}

/**
 * @brief Forward pass for dynamically quantized Linear layer
 * @param input Pointer to floating-point input tensor
 * @param output Pointer to floating-point output tensor
 * 
 * Computes: output = input * dequant(weight)^T + bias
 */
void Linear::forward(float* input, float* output) {
    for (uint32_t j = 0; j < this->output_size; j++) {
        output[j] = 0;
        for (uint32_t i = 0; i < this->input_size; i++) {
            #if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
                // 8-bit weights
                output[j] += input[i] * this->weight[(j * this->input_size) + i];
            #elif QUANTIZATION_BITWIDTH == 4
                // 4-bit weights (packed 2 values per byte)
                output[j] += input[i] * 
                    ((int8_t)(((this->weight[((j * this->input_size) + i)>>1] >> 
                    ((((j * this->input_size) + i) & 1) << 2)) & 0x0F) << 4) >> 4) * 
                    this->weight_scale;
            #endif // QUANTIZATION_BITWIDTH
        }
        output[j] *= this->weight_scale;
        if (this->bias) {
            output[j] +=  this->bias[j];
        }
    }
}

#elif defined(STATIC_QUANTIZATION_PER_TENSOR)

Linear::Linear(uint32_t output_size, uint32_t input_size, const int8_t* weight, const int32_t* bias,
          float output_scale, int8_t output_zero_point, int8_t input_zero_point,  float bias_scale) {

    this->output_size = output_size;
    this->input_size = input_size;

    this->weight = weight;
    this->bias = bias;

    this->output_scale = output_scale;
    this->output_zero_point = output_zero_point;
    this->input_zero_point = input_zero_point;
    
    this->bias_scale = bias_scale;
}

/**
 * @brief Forward pass for statically quantized Linear layer
 * @param input Pointer to quantized input tensor (int8_t)
 * @param output Pointer to quantized output tensor (int8_t)
 * 
 * Computes quantized output with proper scaling and zero-point adjustments
 */
void Linear::forward(int8_t* input, int8_t* output) {
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
        
        // Requantize to 8-bit
        output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
        output_temp += this->output_zero_point;
        
        // Clamp to int8_t range
        output[j] = output_temp < -128 ? (int8_t)-128 : 
                   (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);
    }
}


#else   

/**
 * @brief Constructor for floating-point Linear layer
 * @param output_size Number of output neurons
 * @param input_size Number of input features
 * @param weight Pointer to weight matrix (row-major, shape [output_size, input_size])
 * @param bias Pointer to bias vector (size [output_size])
 */
Linear::Linear(uint32_t output_size, uint32_t input_size, 
              const float* weight, const float* bias) {
    this->output_size = output_size;
    this->input_size = input_size;
    this->weight = weight;
    this->bias = bias;
}

/**
 * @brief Forward pass for floating-point Linear layer
 * @param input Pointer to input tensor (float)
 * @param output Pointer to output tensor (float)
 * 
 * Computes: output = input * weight^T + bias
 */
void Linear::forward(float* input, float* output) {
    for (uint32_t j = 0; j < this->output_size; j++) {
        if (this->bias) {output[j] =  this->bias[j];}
        else {output[j] =  0;}
        // Matrix-vector multiplication
        for (uint32_t i = 0; i < this->input_size; i++) {
            output[j] += input[i] * this->weight[(j * this->input_size) + i];
        }
    }
}


#endif // QUANTIZATION_TYPE

// #if defined(QUANTIZATION_NONE) || (!defined(DYNAMIC_QUANTIZATION_PER_TENSOR) && !defined(DYNAMIC_QUANTIZATION_PER_CHANNEL) \
//                                && !defined(STATIC_QUANTIZATION_PER_TENSOR) && !defined(STATIC_QUANTIZATION_PER_CHANNEL))


// #elif defined(DYNAMIC_QUANTIZATION_PER_TENSOR)

// /**
//  * @brief Constructor for dynamically quantized Linear layer
//  * @param output_size Number of output neurons
//  * @param input_size Number of input features
//  * @param weight Pointer to quantized weight matrix (int8_t)
//  * @param weight_scale Scaling factor for weights
//  * @param bias Pointer to floating-point bias vector
//  */
// Linear::Linear(uint32_t output_size, uint32_t input_size,
//               const int8_t* weight, float weight_scale,
//               const float* bias) {
//     this->output_size = output_size;
//     this->input_size = input_size;
//     this->weight = weight;
//     this->weight_scale = weight_scale;
//     this->bias = bias;
// }

// /**
//  * @brief Forward pass for dynamically quantized Linear layer
//  * @param input Pointer to floating-point input tensor
//  * @param output Pointer to floating-point output tensor
//  * 
//  * Computes: output = input * dequant(weight)^T + bias
//  */
// void Linear::forward(float* input, float* output) {
//     for (uint32_t j = 0; j < this->output_size; j++) {
//         output[j] = 0;
//         for (uint32_t i = 0; i < this->input_size; i++) {
//             #if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
//                 // 8-bit weights
//                 output[j] += input[i] * this->weight[(j * this->input_size) + i] * this->weight_scale;
//             #elif QUANTIZATION_BITWIDTH == 4
//                 // 4-bit weights (packed 2 values per byte)
//                 output[j] += input[i] * 
//                     ((int8_t)(((this->weight[((j * this->input_size) + i)>>1] >> 
//                     ((((j * this->input_size) + i) & 1) << 2)) & 0x0F) << 4) >> 4) * 
//                     this->weight_scale;
//             #endif // QUANTIZATION_BITWIDTH
//         }
//         output[j] += this->bias[j];
//     }
// }

// #elif defined(STATIC_QUANTIZATION_PER_TENSOR)

// /**
//  * @brief Constructor for statically quantized Linear layer
//  * @param output_size Number of output neurons
//  * @param input_size Number of input features
//  * @param output_scale Scaling factor for output
//  * @param output_zero_point Zero point for output
//  * @param input_zero_point Zero point for input
//  * @param weight Pointer to quantized weight matrix (int8_t)
//  * @param bias Pointer to quantized bias vector (int32_t)
//  * @param bias_scale Scaling factor for bias
//  */
// Linear::Linear(uint32_t output_size, uint32_t input_size,
//               float output_scale, int8_t output_zero_point,
//               int8_t input_zero_point, const int8_t* weight,
//               const int32_t* bias, float bias_scale) {
//     this->output_size = output_size;
//     this->input_size = input_size;
//     this->output_scale = output_scale;
//     this->output_zero_point = output_zero_point;
//     this->input_zero_point = input_zero_point;
//     this->weight = weight;
//     this->bias = bias;
//     this->bias_scale = bias_scale;
// }

// /**
//  * @brief Forward pass for statically quantized Linear layer
//  * @param input Pointer to quantized input tensor (int8_t)
//  * @param output Pointer to quantized output tensor (int8_t)
//  * 
//  * Computes quantized output with proper scaling and zero-point adjustments
//  */
// void Linear::forward(int8_t* input, int8_t* output) {
//     int32_t output_temp;

//     for (uint32_t j = 0; j < this->output_size; j++) {
//         output_temp = 0;
//         for (uint32_t i = 0; i < this->input_size; i++) {
//             #if !defined(QUANTIZATION_BITWIDTH) || QUANTIZATION_BITWIDTH == 8
//                 // 8-bit weights
//                 output_temp += ((int32_t)input[i] - this->input_zero_point) * 
//                              (int32_t)this->weight[(j * this->input_size) + i];
//             #elif QUANTIZATION_BITWIDTH == 4
//                 // 4-bit weights (packed 2 values per byte)
//                 output_temp += ((int32_t)input[i] - this->input_zero_point) * 
//                     (int32_t)((int8_t)(((this->weight[((j * this->input_size) + i)>>1] >> 
//                     ((((j * this->input_size) + i) & 1) << 2)) & 0x0F) << 4) >> 4);
//             #endif // QUANTIZATION_BITWIDTH
//         }
//         output_temp += this->bias[j];
        
//         // Requantize to 8-bit
//         output_temp = roundf(output_temp * this->bias_scale / this->output_scale);
//         output_temp += this->output_zero_point;
        
//         // Clamp to int8_t range
//         output[j] = output_temp < -128 ? (int8_t)-128 : 
//                    (output_temp > 127 ? (int8_t)127 : (int8_t)output_temp);
//     }
// }

// #endif // QUANTIZATION_NONE
/**
 * @file sequential.cpp
 * @brief Implementation of sequential neural network model with:
 *      1. Support for both floating-point and quantized inference
 *      2. Double-buffering memory strategy for efficient layer processing
 *      3. Workspace optimization for memory-constrained devices
 * 
 * The implementation uses compile-time switching between floating-point
 * and quantized versions via STATIC_QUANTIZATION_PER_TENSOR define
 */

#include "sequential.h"


#ifdef STATIC_QUANTIZATION_PER_TENSOR // QUANTIZATION_TYPE
Sequential::Sequential(Layer **layers, uint32_t layers_len, int8_t *workspace, 
                      uint32_t workspace_even_layer_size) {
    this->layers = layers;
    this->layers_len = layers_len;
    this->workspace_even_layer = workspace;
    this->workspace_odd_layer = workspace + workspace_even_layer_size;

    // Set model input/output buffers based on double-buffering strategy
    this->input = this->workspace_even_layer;
    this->output = (layers_len % 2 == DLAI_EVEN) ? this->workspace_even_layer 
                                               : this->workspace_odd_layer;   
}

void Sequential::predict(void) {
    for (int i = 0; i < this->layers_len; i++) {
        switch (i % 2) {
            case DLAI_EVEN:
                this->layers[i]->forward(this->workspace_even_layer, 
                                       this->workspace_odd_layer);
                break;
            default:
                this->layers[i]->forward(this->workspace_odd_layer, 
                                       this->workspace_even_layer);
                break;
        }
    }
}
#else // DYNAMIC_QUANTIZATION_PER_TENSOR

/**
 * @brief Constructs a floating-point sequential model
 * @param layers Array of layer pointers
 * @param layers_len Number of layers in model
 * @param workspace Pre-allocated workspace buffer (float)
 * @param workspace_even_layer_size Size of even layer workspace partition
 * 
 * @note Uses double-buffering strategy to alternate between workspace_even_layer
 *       and workspace_odd_layer for memory efficiency
 */
Sequential::Sequential(Layer **layers, uint32_t layers_len, float *workspace, 
                      uint32_t workspace_even_layer_size) {
    this->layers = layers;
    this->layers_len = layers_len;
    this->workspace_even_layer = workspace;
    this->workspace_odd_layer = workspace + workspace_even_layer_size;

    // Set model input/output buffers based on double-buffering strategy
    this->input = this->workspace_even_layer;
    this->output = (layers_len % 2 == DLAI_EVEN) ? this->workspace_even_layer 
                                               : this->workspace_odd_layer;   
}

/**
 * @brief Executes forward pass through all layers
 * 
 * Alternates between workspace buffers for each layer to minimize memory usage:
 * - Even layers write to odd workspace
 * - Odd layers write to even workspace
 */
void Sequential::predict(void) {
    for (int i = 0; i < this->layers_len; i++) {
        switch (i % 2) {
            case DLAI_EVEN:
                this->layers[i]->forward(this->workspace_even_layer, 
                                       this->workspace_odd_layer);
                break;
            default:
                this->layers[i]->forward(this->workspace_odd_layer, 
                                       this->workspace_even_layer);
                break;
        }
    }
}



#endif // QUANTIZATION_TYPE


// #if !defined(STATIC_QUANTIZATION_PER_TENSOR)
// // ==============================================
// // Floating-Point Implementation
// // ==============================================

// #else
// // ==============================================
// // Quantized Implementation (Static Per-Tensor)
// // ==============================================

// /**
//  * @brief Constructs a quantized sequential model
//  * @param layers Array of layer pointers
//  * @param layers_len Number of layers in model
//  * @param workspace Pre-allocated workspace buffer (int8_t)
//  * @param workspace_even_layer_size Size of even layer workspace partition
//  * 
//  * @note Uses same double-buffering strategy as floating-point version
//  *       but with quantized (int8_t) data type
//  */
// Sequential::Sequential(Layer **layers, uint32_t layers_len, int8_t *workspace,
//                       uint32_t workspace_even_layer_size) {
//     this->layers = layers;
//     this->layers_len = layers_len;
//     this->workspace_even_layer = workspace;
//     this->workspace_odd_layer = workspace + workspace_even_layer_size;

//     // Set model input/output buffers based on double-buffering strategy
//     this->input = this->workspace_even_layer;
//     this->output = (layers_len % 2 == DLAI_EVEN) ? this->workspace_even_layer 
//                                                : this->workspace_odd_layer;   
// }

// /**
//  * @brief Executes forward pass through all quantized layers
//  * 
//  * Uses same alternating buffer strategy as floating-point version,
//  * but operates on quantized (int8_t) data
//  */
// void Sequential::predict(void) {
//     for (int i = 0; i < this->layers_len; i++) {
//         switch (i % 2) {
//             case DLAI_EVEN:
//                 this->layers[i]->forward(this->workspace_even_layer, 
//                                        this->workspace_odd_layer);
//                 break;
//             default:
//                 this->layers[i]->forward(this->workspace_odd_layer, 
//                                        this->workspace_even_layer);
//                 break;
//         }
//     }
// }

// #endif // STATIC_QUANTIZATION_PER_TENSOR
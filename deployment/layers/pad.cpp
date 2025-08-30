#include "pad.h"
#include "layer.h"

void pad_input(float* input, Padding_t padding, 
                const uint32_t input_channel_size, const uint32_t input_row_size, const uint32_t input_col_size, 
                const uint32_t padded_row_size, const uint32_t padded_col_size) {
    if (padding.is_padded()) {
        for (int32_t n = input_channel_size-1; n > -1; n--) {
            for (int32_t m = padded_row_size-1; m > -1; m--) {
                for (int32_t l = padded_col_size-1; l > -1; l--) {

                    if (m < padding.padding_top || m >= padded_row_size - padding.padding_bottom || 
                        l < padding.padding_left || l >= padded_col_size - padding.padding_right){
                        
                            input[((n * padded_row_size * padded_col_size) + 
                            (m * padded_col_size) + 
                            l)] = 0;
                        }
                    else {
                            input[((n * padded_row_size * padded_col_size) + 
                            (m * padded_col_size) + 
                            l)] =
                            input[((n * input_row_size * input_col_size) + 
                            ((m-padding.padding_top) * input_col_size) + 
                            (l-padding.padding_left))];
                    }
                }
            }
        }

    }
}


void pad_input(int8_t* input, int8_t zero_point, Padding_t padding, 
                const uint32_t input_channel_size, const uint32_t input_row_size, const uint32_t input_col_size, 
                const uint32_t padded_row_size, const uint32_t padded_col_size) {
    if (padding.is_padded()) {
        for (int32_t n = input_channel_size-1; n > -1; n--) {
            for (int32_t m = padded_row_size-1; m > -1; m--) {
                for (int32_t l = padded_col_size-1; l > -1; l--) {

                    // if (m < padding.padding_top || m >= padded_row_size - padding.padding_bottom || 
                    //     l < padding.padding_left || l >= padded_col_size - padding.padding_right){
                        
                    //         input[((n * padded_row_size * padded_col_size) + 
                    //         (m * padded_col_size) + 
                    //         l)] = zero_point;
                    //     }
                    // else {
                    //         input[((n * padded_row_size * padded_col_size) + 
                    //         (m * padded_col_size) + 
                    //         l)] =
                    //         input[((n * input_row_size * input_col_size) + 
                    //         ((m-padding.padding_top) * input_col_size) + 
                    //         (l-padding.padding_left))];
                    // }

                    if (m < padding.padding_top || m >= padded_row_size - padding.padding_bottom || 
                        l < padding.padding_left || l >= padded_col_size - padding.padding_right){
                        
                            set_packed_value(input, 
                                ((n * padded_row_size * padded_col_size) + 
                                (m * padded_col_size) + 
                                l),
                                zero_point
                            );
                            // input[((n * padded_row_size * padded_col_size) + 
                            // (m * padded_col_size) + 
                            // l)] = zero_point;
                        }
                    else {
                            set_packed_value(input,
                                ((n * padded_row_size * padded_col_size) + 
                                (m * padded_col_size) + 
                                l),
                                get_packed_value(input,
                                    ((n * input_row_size * input_col_size) + 
                                    ((m-padding.padding_top) * input_col_size) + 
                                    (l-padding.padding_left))
                                )
                            );
                    //         input[((n * padded_row_size * padded_col_size) + 
                    //         (m * padded_col_size) + 
                    //         l)] =
                    //         input[((n * input_row_size * input_col_size) + 
                    //         ((m-padding.padding_top) * input_col_size) + 
                    //         (l-padding.padding_left))];
                    }
                }
            }
        }
    }
}


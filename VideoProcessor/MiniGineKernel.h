#pragma once

void mini_gine_draw_entities_kernel_launch(const unsigned int* device_data_assets, const unsigned int models_position,
    const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int entities_count,
    unsigned char *device_data_output, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int tick_counter);
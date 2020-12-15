#include "MiniGineS.h"
#include "MiniGineKernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDAStreamHandler.h"
#include "Grid.h"

__forceinline__
__device__ float getInterpixel(const unsigned char* frame, const unsigned int width, const unsigned int height, const unsigned int channels, float x, float y, const int c) {
    int x_i = (int)x;
    int y_i = (int)y;
    x -= x_i;
    y -= y_i;

    unsigned char value_components[4];
    value_components[0] = frame[y_i * (width * channels) + x_i * channels + c];
    if (x > 0) {
        if (x_i + 1 < width) {
            value_components[1] = frame[y_i * (width * channels) + (x_i + 1) * channels + c];
        }
        else {
            x = 0.0f;
        }
    }
    if (y > 0) {
        if (y_i + 1 < height) {
            value_components[2] = frame[(y_i + 1) * (width * channels) + x_i * channels + c];
            if (x > 0) {
                value_components[3] = frame[(y_i + 1) * (width * channels) + (x_i + 1) * channels + c];
            }
        }
        else {
            y = 0.0f;
        }
    }

    float m_0 = 4.0f / 16.0f;
    float m_1 = 4.0f / 16.0f;
    float m_2 = 4.0f / 16.0f;
    float m_3 = 4.0f / 16.0f;
    float tmp, tmp2;
    if (x <= 0.5f) {
        tmp = ((0.5f - x) / 0.5f) * m_1;
        m_0 += tmp;
        m_1 -= tmp;
        m_2 += tmp;
        m_3 -= tmp;
    }
    else {
        tmp = ((x - 0.5f) / 0.5f) * m_0;
        m_0 -= tmp;
        m_1 += tmp;
        m_2 -= tmp;
        m_3 += tmp;
    }
    if (y <= 0.5f) {
        tmp = ((0.5f - y) / 0.5f) * m_2;
        tmp2 = ((0.5f - y) / 0.5f) * m_3;
        m_0 += tmp;
        m_1 += tmp2;
        m_2 -= tmp;
        m_3 -= tmp2;
    }
    else {
        tmp = ((y - 0.5f) / 0.5f) * m_0;
        tmp2 = ((y - 0.5f) / 0.5f) * m_1;
        m_0 -= tmp;
        m_1 -= tmp2;
        m_2 += tmp;
        m_3 += tmp2;
    }
    float value = m_0 * value_components[0] + m_1 * value_components[1] + m_2 * value_components[2] + m_3 * value_components[3];
    return value;
}

__global__ void mini_gine_draw_entities_kernel(
    const unsigned int* device_data_assets, const unsigned int models_position,
    const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int gd_position_in_bf, const unsigned int gd_data_position_in_bf,
    unsigned char *device_data_output, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int tick_counter) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    struct mini_gine_entity* entities = (struct mini_gine_entity*)&device_data_rw[entities_position];   
    struct mini_gine_model* models = (struct mini_gine_model*)&device_data_assets[models_position];

    if (i < output_width * output_height * output_channels) {
        int current_channel = i / (output_width * output_height);
        int current_idx = i % (output_width * output_height);
        int current_x = (current_idx % output_width);
        int current_y = (current_idx / output_width);

        unsigned char* output = device_data_output;

        int grid_current_idx = grid_get_index(device_data_rw, gd_position_in_bf, struct vector3<float>(current_x, current_y, 0.0f));
        if (grid_current_idx != -1) {
            unsigned int entities_iddata_position = device_data_rw[gd_data_position_in_bf + 1 + grid_current_idx];
            if (entities_iddata_position > 0) {
                unsigned int entities_count = device_data_rw[entities_iddata_position];

                unsigned int player_z_max = 0;
                float player_y_max = -1.0f;
                int player_id_max = -1;

                for (int e = 0; e < entities_count; e++) {
                    unsigned int entity_id = device_data_rw[entities_iddata_position + 1 + e];
                    if (entity_id < UINT_MAX) {
                        struct mini_gine_model* m = &models[entities[entity_id].model_id];
                        const unsigned int* model_positions_in_bf = &device_data_assets[m->model_positions];

                        float sampling_filter_dim = ceilf(1.0f / entities[entity_id].scale);

                        float offset_to_model_base_x = (current_x - (entities[entity_id].position[0])) / entities[entity_id].scale;
                        float offset_to_model_base_y = (current_y - (entities[entity_id].position[1])) / entities[entity_id].scale;

                        if (offset_to_model_base_x >= entities[entity_id].crop_x[0] && offset_to_model_base_x < entities[entity_id].crop_x[1] &&
                            offset_to_model_base_y >= entities[entity_id].crop_y[0] && offset_to_model_base_y < entities[entity_id].crop_y[1]) {

                            int animation_tick = (((tick_counter + entities[entity_id].model_animation_offset) / m->model_animation_stepsize) % m->model_animation_ticks);
                            if (m->model_animation_type == 1) {
                                if (((tick_counter + entities[entity_id].model_animation_offset) / m->model_animation_stepsize) / m->model_animation_ticks % 2 == 1) {
                                    animation_tick = m->model_animation_ticks - 1 - animation_tick;
                                }
                            }

                            int model_idx = ((int)(entities[entity_id].orientation / 10 / (36 / m->model_rotations)) % m->model_rotations) * m->model_animation_ticks + animation_tick;
                            if (m->model_params > 0) {
                                model_idx = ((int)(entities[entity_id].orientation / 10 / (36 / m->model_rotations)) % m->model_rotations) * m->model_animation_ticks * (m->model_params + 2) + animation_tick * (m->model_params + 2);

                                const unsigned char* zero_m = (unsigned char*)&device_data_assets[model_positions_in_bf[model_idx]];
                                const unsigned char* full_m = (unsigned char*)&device_data_assets[model_positions_in_bf[model_idx + 1 + m->model_params]];

                                struct mini_gine_model_params* mgmp = (struct mini_gine_model_params*)&device_data_rw[entities[entity_id].model_params_position];

                                float zero = getInterpixel(zero_m, m->model_dimensions[0], m->model_dimensions[1], 4, offset_to_model_base_x, offset_to_model_base_y, current_channel);
                                float interpixel_alpha = getInterpixel(zero_m, m->model_dimensions[0], m->model_dimensions[1], 4, offset_to_model_base_x, offset_to_model_base_y, 3);

                                float full = getInterpixel(full_m, m->model_dimensions[0], m->model_dimensions[1], 4, offset_to_model_base_x, offset_to_model_base_y, current_channel);

                                float value = full - zero;

                                float weight_norm = 0.0f;
                                float total_weight = 0.0f;
                                for (int dims = 0; dims < m->model_params; dims++) {
                                    const unsigned char* dim = (unsigned char*)&device_data_assets[model_positions_in_bf[model_idx + 1 + dims]];
                                    float weight = 0.0f;
                                    if (abs(value) < 1) {

                                    } else {
                                        float dim_v = getInterpixel(dim, m->model_dimensions[0], m->model_dimensions[1], 4, offset_to_model_base_x, offset_to_model_base_y, current_channel);
                                        weight = (dim_v - zero) / value;
                                        if (current_channel == 0) {
                                            total_weight += weight * (mgmp->b / 255.0f) * mgmp->s;
                                        }
                                        else if (current_channel == 1) {
                                            total_weight += weight * (mgmp->g / 255.0f) * mgmp->s;
                                        }
                                        else if (current_channel == 2) {
                                            total_weight += weight * (mgmp->r / 255.0f) * mgmp->s;
                                        }
                                    }
                                    weight_norm += weight;

                                    mgmp++;
                                }
                                if (weight_norm > 0.0f) {
                                    total_weight /= weight_norm;
                                    float output_v = zero + total_weight * value;
                                    if (e == 0) {
                                        output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = output_v;
                                    } else {
                                        output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (interpixel_alpha / 255.0f) * output_v));
                                    }
                                } else {
                                    if (e == 0) {
                                        output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = zero;
                                    } else {
                                        output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (interpixel_alpha / 255.0f) * zero));
                                    }
                                }
                            } else {
                                unsigned char* p_model = (unsigned char*)&device_data_assets[model_positions_in_bf[model_idx]];

                                for (int s_y = 0; s_y < sampling_filter_dim; s_y++) {
                                    for (int s_x = 0; s_x < sampling_filter_dim; s_x++) {
                                        if (offset_to_model_base_x + s_x >= entities[entity_id].crop_x[0] && offset_to_model_base_x + s_x < entities[entity_id].crop_x[1] &&
                                            offset_to_model_base_y + s_y >= entities[entity_id].crop_y[0] && offset_to_model_base_y + s_y < entities[entity_id].crop_y[1]
                                            ) {
                                            float model_palette_idx_x = offset_to_model_base_x + s_x;
                                            float model_palette_idx_y = offset_to_model_base_y + s_y;

                                            float interpixel_alpha = getInterpixel(p_model, m->model_dimensions[0], m->model_dimensions[1], 4, model_palette_idx_x, model_palette_idx_y, 3);
                                            if (interpixel_alpha > 0) {
                                                float interpixel = getInterpixel(p_model, m->model_dimensions[0], m->model_dimensions[1], 4, model_palette_idx_x, model_palette_idx_y, current_channel);
                                                if (e == 0) {
                                                    output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)interpixel;
                                                } else {
                                                    output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] = (unsigned char)(((255 - interpixel_alpha) / 255.0f * output[current_y * (output_width * output_channels) + current_x * output_channels + current_channel] + (interpixel_alpha / 255.0f) * interpixel));
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                        }
                    }
                }
            }
        }     
    }
}

void mini_gine_draw_entities_kernel_launch(const unsigned int* device_data_assets, const unsigned int models_position,
    const unsigned int* device_data_rw, const unsigned int entities_position, const unsigned int gd_position_in_bf, const unsigned int gd_data_position_in_bf,
    unsigned char* device_data_output, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int tick_counter) {

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_width * output_height * output_channels + threadsPerBlock - 1) / threadsPerBlock;
    
    mini_gine_draw_entities_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[1] >> > (device_data_assets, models_position, device_data_rw, entities_position, gd_position_in_bf, gd_data_position_in_bf,
        device_data_output, output_width, output_height, output_channels, tick_counter);
    cudaStreamSynchronize(cuda_streams[1]);
}
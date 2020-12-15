#pragma once

#include <cuda_runtime.h>
#include "BitField.h"
#include "Vector3.h"
#include "string.h"

#define SIZEOF_GRID 44
#define SIZEOF_GRID_IN_BF 11

struct grid {
    unsigned int 		position_in_bf;

    vector3<int> 		scaled_dimensions;
    vector3<float>		scale;
    vector3<float>		center;

    unsigned int		data_position_in_bf;
};

__forceinline__
__host__ void grid_init(struct bit_field* bf, struct grid* out_grid, const vector3<float> dimensions, const vector3<float> scale, const vector3<float> center) {
    vector3<unsigned int> total_dim = { (unsigned int)ceil(dimensions[0] / scale[0]), (unsigned int)ceil(dimensions[1] / scale[1]), (unsigned int)ceil(dimensions[2] / scale[2]) };
    out_grid->scaled_dimensions = { (int)total_dim[0], (int)total_dim[1], (int)total_dim[2] };
    out_grid->scale = scale;
    out_grid->center = center;

    unsigned int total_size_in_bf = (total_dim[2] * (total_dim[1] * total_dim[0]) + total_dim[1] * total_dim[1] + total_dim[0]);

    out_grid->data_position_in_bf = bit_field_add_bulk_zero(bf, total_size_in_bf);
    out_grid->position_in_bf = bit_field_add_bulk(bf, (unsigned int*)out_grid, SIZEOF_GRID_IN_BF, SIZEOF_GRID);
}

__forceinline__
__host__ __device__ int grid_get_index(const unsigned int* bf_data, const unsigned int position_in_bf, const vector3<float> position) {
    unsigned int gd_tmp[SIZEOF_GRID_IN_BF];
    memcpy(&gd_tmp, &bf_data[position_in_bf + 1], SIZEOF_GRID);
    struct grid* gd = (struct grid*)&gd_tmp;

    vector3<int> grid_vec = { (int)(floor(position[0] + gd->center[0]) / gd->scale[0]), (int)(floor(position[1] + gd->center[1]) / gd->scale[1]), (int)(floor(position[2] + gd->center[2]) / gd->scale[2]) };
    if (grid_vec[0] < 0 || grid_vec[0] >= gd->scaled_dimensions[0] || grid_vec[1] < 0 || grid_vec[1] >= gd->scaled_dimensions[1] || grid_vec[2] < 0 || grid_vec[2] >= gd->scaled_dimensions[2]) {
        return -1;
    }
    return (int)(grid_vec[2] * (gd->scaled_dimensions[1] * gd->scaled_dimensions[0]) + grid_vec[1] * gd->scaled_dimensions[0] + grid_vec[0]);
}

__forceinline__
__host__ __device__ int grid_get_max_index(const unsigned int* bf_data, const unsigned int position_in_bf) {
    unsigned int gd_tmp[SIZEOF_GRID_IN_BF];
    memcpy(&gd_tmp, &bf_data[position_in_bf + 1], SIZEOF_GRID);
    struct grid* gd = (struct grid*)&gd_tmp;

    return grid_get_index(bf_data, position_in_bf, { (float)(gd->scaled_dimensions[0] - 1), (float)(gd->scaled_dimensions[1] - 1), (float)(gd->scaled_dimensions[2] - 1) });
}

__forceinline__
__host__ __device__ float grid_traverse_in_direction(const unsigned int* bf_data, const unsigned int position_in_bf, const vector3<float> position, const vector3<float> direction) {
    unsigned int gd_tmp[SIZEOF_GRID_IN_BF];
    memcpy(&gd_tmp, &bf_data[position_in_bf + 1], SIZEOF_GRID);
    struct grid* gd = (struct grid*)&gd_tmp;

    vector3<int> grid_vec = { (int)floor(position[0] + gd->center[0]), (int)floor(position[1] + gd->center[1]), (int)floor(position[2] + gd->center[2]) };

    float lambda_x = sqrtf(3) * gd->scale[0];
    int x_dir = 0;
    if (direction[0] != 0) {
        if (direction[0] > 0) {
            x_dir = 1;
        }
        else {
            x_dir = -1;
        }
        lambda_x = (grid_vec[0] + x_dir - position[0] - gd->center[0]) / (direction[0]);
    }
    float lambda_y = sqrtf(3) * gd->scale[1];
    int y_dir = 0;
    if (direction[1] != 0) {
        if (direction[1] > 0) {
            y_dir = 1;
        }
        else {
            y_dir = -1;
        }
        lambda_y = (grid_vec[1] + y_dir - position[1] - gd->center[1]) / (direction[1]);
    }
    float lambda_z = sqrtf(3) * gd->scale[2];
    int z_dir = 0;
    if (direction[2] != 0) {
        if (direction[2] > 0) {
            z_dir = 1;
        }
        else {
            z_dir = -1;
        }
        lambda_z = (grid_vec[2] + z_dir - position[2] - gd->center[2]) / (direction[2]);
    }
    if (lambda_x <= lambda_y && lambda_x <= lambda_z) {
        return lambda_x;
    }
    if (lambda_y <= lambda_x && lambda_y <= lambda_z) {
        return lambda_y;
    }
    if (lambda_z <= lambda_x && lambda_z <= lambda_y) {
        return lambda_z;
    }
    return 0.0f;
}

__forceinline__
__host__ __device__ int grid_object_add(struct bit_field* bf, unsigned int* bf_data, const unsigned int position_in_bf, const vector3<float> position, const vector3<float> scale, const vector3<float> position_min, const vector3<float> position_max, const unsigned int id) {
    unsigned int gd_tmp[SIZEOF_GRID_IN_BF];
    memcpy(&gd_tmp, &bf_data[position_in_bf + 1], SIZEOF_GRID);
    struct grid* gd = (struct grid*)&gd_tmp;
    int x_neighbour_start = (int)floor((position[0] + (position_min[0] * scale[0]) + gd->center[0])) / gd->scale[0];
    int y_neighbour_start = (int)floor((position[1] + (position_min[1] * scale[1]) + gd->center[1])) / gd->scale[1];
    int z_neighbour_start = (int)floor((position[2] + (position_min[2] * scale[2]) + gd->center[2])) / gd->scale[2];
    int x_neighbour_end = (int)ceil((position[0] + (position_max[0] * scale[0]) + gd->center[0])) / gd->scale[0];
    int y_neighbour_end = (int)ceil((position[1] + (position_max[1] * scale[1]) + gd->center[1])) / gd->scale[1];
    int z_neighbour_end = (int)ceil((position[2] + (position_max[2] * scale[2]) + gd->center[2])) / gd->scale[2];
    for (int k = z_neighbour_start; k <= z_neighbour_end; k++) {
        for (int j = y_neighbour_start; j <= y_neighbour_end; j++) {
            for (int i = x_neighbour_start; i <= x_neighbour_end; i++) {
                //					printf("grid add %d: %d %d %d\n", id, i, j, k);
                if (i < 0 || i >= gd->scaled_dimensions[0]) continue;
                if (j < 0 || j >= gd->scaled_dimensions[1]) continue;
                if (k < 0 || k >= gd->scaled_dimensions[2]) continue;
                //printf("i %d,j %d, k %d", i, j, k);
                unsigned int cur_idx = k * (gd->scaled_dimensions[1] * gd->scaled_dimensions[0]) + j * (gd->scaled_dimensions[0]) + i;
                bf_data = bf->data;
                unsigned int cur_val = bf_data[gd->data_position_in_bf + 1 + cur_idx];
                if (cur_val == 0) {
#ifdef CUDA_ARCH
                    //TODO: solve
#else
                    bit_field_update_data(bf, gd->data_position_in_bf + 1 + cur_idx, bit_field_add_data(bf, id));
#endif
                }
                else {
#ifdef CUDA_ARCH
                    //TODO: solve
#else
                    bool found = 0;
                    for (int s = 0; s < bf_data[cur_val]; s++) {
                        if (bf_data[cur_val + 1 + s] == id) found = 1;
                    }
                    if (found == 0) {
                        int added = 0;
                        for (int s = 0; s < bf_data[cur_val]; s++) {
                            if (bf_data[cur_val + 1 + s] == UINT_MAX) {
                                bit_field_update_data(bf, cur_val + 1 + s, id);
                                added = 1;
                                break;
                            }
                        }
                        if (added == 0) {
                            unsigned int check_for_realloc = bit_field_add_data_to_segment(bf, cur_val, id);
                            if (check_for_realloc != cur_val) {
                                bit_field_update_data(bf, gd->data_position_in_bf + 1 + cur_idx, check_for_realloc);
                            }
                        }
                    }
#endif
                }
            }
        }
    }
    return 0;
}

__forceinline__
__host__ __device__ int grid_object_remove(struct bit_field* bf, unsigned int* bf_data, unsigned int position_in_bf, const vector3<float> position, const vector3<float> scale, const vector3<float> position_min, const vector3<float> position_max, const unsigned int id) {
    unsigned int gd_tmp[SIZEOF_GRID_IN_BF];
    memcpy(&gd_tmp, &bf_data[position_in_bf + 1], SIZEOF_GRID);
    struct grid* gd = (struct grid*)&gd_tmp;
    int x_neighbour_start = (int)floor((position[0] + (position_min[0] * scale[0]) + gd->center[0])) / gd->scale[0];
    int y_neighbour_start = (int)floor((position[1] + (position_min[1] * scale[1]) + gd->center[1])) / gd->scale[1];
    int z_neighbour_start = (int)floor((position[2] + (position_min[2] * scale[2]) + gd->center[2])) / gd->scale[2];
    int x_neighbour_end = (int)ceil((position[0] + (position_max[0] * scale[0]) + gd->center[0])) / gd->scale[0];
    int y_neighbour_end = (int)ceil((position[1] + (position_max[1] * scale[1]) + gd->center[1])) / gd->scale[1];
    int z_neighbour_end = (int)ceil((position[2] + (position_max[2] * scale[2]) + gd->center[2])) / gd->scale[2];
    for (int k = z_neighbour_start; k <= z_neighbour_end; k++) {
        for (int j = y_neighbour_start; j <= y_neighbour_end; j++) {
            for (int i = x_neighbour_start; i <= x_neighbour_end; i++) {
                if (i < 0 || i >= gd->scaled_dimensions[0]) continue;
                if (j < 0 || j >= gd->scaled_dimensions[1]) continue;
                if (k < 0 || k >= gd->scaled_dimensions[2]) continue;
                unsigned int cur_idx = k * (gd->scaled_dimensions[1] * gd->scaled_dimensions[0]) + j * (gd->scaled_dimensions[0]) + i;
                unsigned int cur_val = bf_data[gd->data_position_in_bf + 1 + cur_idx];
                if (cur_val != 0) {
#ifdef CUDA_ARCH
                    //TODO: solve
#else
                    for (int s = 0; s < bf_data[cur_val]; s++) {
                        if (bf_data[cur_val + 1 + s] == id) {
                            bit_field_update_data(bf, cur_val + 1 + s, UINT_MAX);
                            break;
                        }
                    }
#endif
                }
            }
        }
    }
    return 0;
}

__forceinline__
__host__ void grid_dump(unsigned int* bf_data, unsigned int position_in_bf) {
    struct grid* gd = (struct grid*)&bf_data[position_in_bf + 1];
    for (int z = 0; z < gd->scaled_dimensions[2]; z++) {
        printf("layer: %d\n", z);
        for (int y = 0; y < gd->scaled_dimensions[1]; y++) {
            for (int x = 0; x < gd->scaled_dimensions[0]; x++) {
                unsigned int cur_idx = grid_get_index(bf_data, position_in_bf, { (float)x, (float)y, (float)z });
                unsigned int link = bf_data[gd->data_position_in_bf + 1 + cur_idx];
                if (link != 0) {
                    unsigned int element_count = bf_data[link];
                    for (int c = 0; c < element_count; c++) {
                        printf("%u", bf_data[link + 1 + c]);
                        if (c < element_count - 1) printf(",");
                    }
                }
                printf("\t");
            }
            printf("\n");
        }
    }
}
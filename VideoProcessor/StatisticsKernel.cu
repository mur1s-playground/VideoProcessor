#include "StatisticsKernel.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

#include "Vector2.h"
#include "Logger.h"

__global__ void statistics_heatmap_kernel(float *data, struct vector3<int> dimensions, float falloff) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < dimensions[0] * dimensions[1] * dimensions[2]) {
		data[i] = data[i] * falloff;
	}
}

void statistics_heatmap_kernel_launch(float *data, float *device_data, struct vector3<int> dimensions, float falloff) {
	int size = dimensions[0] * dimensions[1] * dimensions[2];

	cudaMemcpyAsync(device_data, data, size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaStreamSynchronize(cuda_streams[0]);

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	statistics_heatmap_kernel <<<blocksPerGrid, threadsPerBlock, 0, cuda_streams[3]>>> (device_data, dimensions, falloff);
	cudaStreamSynchronize(cuda_streams[3]);

	cudaMemcpyAsync(data, device_data, size * sizeof(float), cudaMemcpyDeviceToHost, cuda_streams[4]);
	cudaStreamSynchronize(cuda_streams[4]);
}


__global__ void statistics_3d_kernel(const float* heatmap_data, const float* vectorfield_data, const float max_vel, const float max_acc, unsigned char* dst, const int width, const int height, struct vector3<int> heatmap_dims, int z_axis) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / width;
		int col = (i % width);

		int screen_height = (height / 2);
		int screen = row / screen_height;
		float screen_ratio = width / (float)screen_height;

		dst[row * width * 3 + col * 3 + 0] = 0;
		dst[row * width * 3 + col * 3 + 1] = 0;
		dst[row * width * 3 + col * 3 + 2] = 0;

		struct vector2<float> world_size_x = { 100000.0f, -100000.0f };
		struct vector2<float> world_size_y = { 100000.0f, -100000.0f };
		struct vector2<float> world_size_z = { 100000.0f, -100000.0f };

		world_size_x = { 0.0f, 30.0f * 20.0f };
		world_size_y = { 0.0f, 30.0f * 20.0f };

		float scaling_factor = 20.0f;

		world_size_x[0] -= (0.2f * max(100.0f, (world_size_x[1] - world_size_x[0])));
		world_size_x[1] += (0.2f * max(100.0f, (world_size_x[1] - world_size_x[0])));

		world_size_y[0] -= (0.2f * max(100.0f, (world_size_y[1] - world_size_y[0])));
		world_size_y[1] += (0.2f * max(100.0f, (world_size_y[1] - world_size_y[0])));

		world_size_z[0] -= (0.2f * max(100.0f, (world_size_z[1] - world_size_z[0])));
		world_size_z[1] += (0.2f * max(100.0f, (world_size_z[1] - world_size_z[0])));

		struct vector3<float> world_dim = { world_size_x[1] - world_size_x[0], world_size_y[1] - world_size_y[0], world_size_z[1] - world_size_z[0] };

		float world_ratio_xy = world_dim[0] / world_dim[1];
		if (world_ratio_xy > screen_ratio) {
			world_size_y[0] = world_size_y[0] / world_ratio_xy * screen_ratio;
			world_size_y[1] = world_size_y[1] / world_ratio_xy * screen_ratio;
		}
		else {
			float world_dim_x_diff = (screen_ratio * world_dim[1]) - world_dim[0];

			world_size_x[0] = world_size_x[0] - (world_dim_x_diff / 2.0f);
			world_size_x[1] = world_size_x[1] + (world_dim_x_diff / 2.0f);

			world_dim[0] = world_size_x[1] - world_size_x[0];

			float world_ratio_xz = world_dim[0] / world_dim[2];
			if (world_ratio_xz > screen_ratio) {
				float world_dim_z_diff = (screen_ratio * world_dim[1]) - world_dim[2];

				world_size_z[0] -= (world_dim_z_diff / 2.0f);
				world_size_z[1] += (world_dim_z_diff / 2.0f);
			}
		}

		world_dim = { world_size_x[1] - world_size_x[0], world_size_y[1] - world_size_y[0], world_size_z[1] - world_size_z[0] };

		float pixel_width = world_dim[0] / (float)width;

		if (abs(row - screen * screen_height) < 2 * pixel_width) {
			dst[row * width * 3 + col * 3 + 0] = 255;
			dst[row * width * 3 + col * 3 + 1] = 255;
			dst[row * width * 3 + col * 3 + 2] = 255;
		}

		if (screen == 0) {
			if (col / (float)width < 0.5) {
				//left sub screen
				struct vector2<int> position = {
					(int)(col / (float)width * 2.0f * heatmap_dims[0]),
					(int)(((((screen + 1) * screen_height - row) % screen_height) / (float)(screen_height)) * (heatmap_dims[1]))
				};

				int idx = position[0] * heatmap_dims[1] * heatmap_dims[2] + position[1] * heatmap_dims[2] + ((z_axis/3) % heatmap_dims[2]);

				float tmp = heatmap_data[idx];
				/*
				float tmp = 0.0f;
				for (int z = 0; z < heatmap_dims[2]; z++) {
					tmp += heatmap_data[idx + z];
				}
				*/
				if (tmp > 1) tmp = 1.0f;
				if (tmp < 0) tmp = 0.0f;

				dst[row * width * 3 + col * 3 + 0] = 0;
				dst[row * width * 3 + col * 3 + 1] = 0;
				dst[row * width * 3 + col * 3 + 2] = 0;

				if (tmp < 0.33) {
					dst[row * width * 3 + col * 3 + 1] = (int)((tmp / 0.33f) * 254);
				}
				else if (tmp < 0.66) {
					dst[row * width * 3 + col * 3 + 0] = (int)((tmp / 0.66f) * 254);
				}
				else {
					dst[row * width * 3 + col * 3 + 2] = (int)(tmp * 254);
				}
			} else {
				//right sub screen
				float x = ((col / (float)width - 0.5f) * 2.0f * heatmap_dims[0]);
				float y = (((((screen + 1) * screen_height - row) % screen_height) / (float)(screen_height)) * (heatmap_dims[1]));

				int x_1, y_1, z_1;

				if (x - floorf(x) < 0.33f) {
					x_1 = 0;
				}
				else if (x - floorf(x) < 0.66f) {
					x_1 = 1;
				}
				else {
					x_1 = 2;
				}

				if (y - floorf(y) < 0.33f) {
					y_1 = 0;
				}
				else if (y - floorf(y) < 0.66f) {
					y_1 = 1;
				}
				else {
					y_1 = 2;
				}

				struct vector2<int> position = {
					(int)x,
					(int)y
				};

				int idx = position[0] * heatmap_dims[1] * heatmap_dims[2] + position[1] * heatmap_dims[2] + ((z_axis / 3) % heatmap_dims[2]);
				idx *= 27 * 3;

				dst[row * width * 3 + col * 3 + 0] = 0;
				dst[row * width * 3 + col * 3 + 1] = 0;
				dst[row * width * 3 + col * 3 + 2] = 0;

				float tmp = 0.0f;

				int z = z_axis % 3;
				//for (int z = 0; z < 3; z++) {
					int inner_idx = (z * 9 + y_1 * 3 + x_1) * 3;
					tmp += vectorfield_data[idx + inner_idx];
				//}
				
				if (tmp == 1.0f / 27.0f) {

				} else if (tmp < 1.0f / 54.0f) {
					dst[row * width * 3 + col * 3 + 0] = (int)(tmp / (1.0f / 54.0f) * 254);
				} else if (tmp < 3.0f / 54.0f) {
					dst[row * width * 3 + col * 3 + 0] = 254 + (((tmp - (1.0f / 54.0f)) / ((3.0f / 54.0f) - (1.0f / 54.0f))) * (0 - 254));
					dst[row * width * 3 + col * 3 + 1] = (int)((tmp - (1.0f / 54.0f)) / (3.0f / 54.0f - 1.0f / 54.0f) * 254);
				} else {
					dst[row * width * 3 + col * 3 + 1] = 254 + (((tmp - (3.0f / 54.0f))/(1 - (3.0f / 54.0f))) * (0 - 254));
					dst[row * width * 3 + col * 3 + 2] = (int)((tmp - (3.0f / 54.0f)) / (1 - (3.0f / 54.0f)) * 254);
				}
			}
		} else if (screen == 1) {
			if (col / (float)width < 0.5) {
				//left sub screen
				float x = ((col / (float)width) * 2.0f * heatmap_dims[0]);
				float y = (((((screen + 1) * screen_height - row) % screen_height) / (float)(screen_height)) * (heatmap_dims[1]));

				int x_1, y_1, z_1;

				if (x - floorf(x) < 0.33f) {
					x_1 = 0;
				} else if (x - floorf(x) < 0.66f) {
					x_1 = 1;
				} else {
					x_1 = 2;
				}

				if (y - floorf(y) < 0.33f) {
					y_1 = 0;
				} else if (y - floorf(y) < 0.66f) {
					y_1 = 1;
				} else {
					y_1 = 2;
				}

				struct vector2<int> position = {
					(int)x,
					(int)y
				};

				int idx = position[0] * heatmap_dims[1] * heatmap_dims[2] + position[1] * heatmap_dims[2] + ((z_axis / 3) % heatmap_dims[2]);
				idx *= 27 * 3;

				dst[row * width * 3 + col * 3 + 0] = 0;
				dst[row * width * 3 + col * 3 + 1] = 0;
				dst[row * width * 3 + col * 3 + 2] = 0;

				float tmp = 0.0f;

				int z = z_axis % 3;
				//for (int z = 0; z < 3; z++) {
					int inner_idx = (z * 9 + y_1 * 3 + x_1) * 3;
					tmp += vectorfield_data[idx + inner_idx + 1];
				//}

				if (tmp < 1.0f / 3.0f * max_vel) {
					dst[row * width * 3 + col * 3 + 0] = (int)(tmp / (1.0f / 3.0f * max_vel) * 254);
				} else if (tmp < 2.0f / 3.0f * max_vel) {			
					dst[row * width * 3 + col * 3 + 0] = (int)(254 + ((tmp - 1.0f / 3.0f * max_vel) / (2.0f / 3.0f * max_vel - (1.0f / 3.0f * max_vel))) * (0 - 254));
					dst[row * width * 3 + col * 3 + 1] = (int)((tmp - (1.0f / 3.0f * max_vel)) / ((2.0f / 3.0f * max_vel) - (1.0f / 3.0f * max_vel)) * 254);
				} else {
					dst[row * width * 3 + col * 3 + 1] = (int)(254 + ((tmp - (2.0f / 3.0f * max_vel)) / (max_vel - (2.0f / 3.0f * max_vel))) * (0 - 254));
					dst[row * width * 3 + col * 3 + 2] = (int)(((tmp - (2.0f / 3.0f * max_vel)) / (max_vel - (2.0f / 3.0f * max_vel))) * 254);
				}
			} else {
				//right sub screen
				float x = ((col / (float)width - 0.5f) * 2.0f * heatmap_dims[0]);
				float y = (((((screen + 1) * screen_height - row) % screen_height) / (float)(screen_height)) * (heatmap_dims[1]));

				int x_1, y_1, z_1;

				if (x - floorf(x) < 0.33f) {
					x_1 = 0;
				} else if (x - floorf(x) < 0.66f) {
					x_1 = 1;
				} else {
					x_1 = 2;
				}

				if (y - floorf(y) < 0.33f) {
					y_1 = 0;
				} else if (y - floorf(y) < 0.66f) {
					y_1 = 1;
				} else {
					y_1 = 2;
				}

				struct vector2<int> position = {
					(int)x,
					(int)y
				};

				int idx = position[0] * heatmap_dims[1] * heatmap_dims[2] + position[1] * heatmap_dims[2] + ((z_axis / 3) % heatmap_dims[2]);
				idx *= 27 * 3;

				dst[row * width * 3 + col * 3 + 0] = 0;
				dst[row * width * 3 + col * 3 + 1] = 0;
				dst[row * width * 3 + col * 3 + 2] = 0;

				float tmp = 0.0f;

				int z = z_axis % 3;
				//for (int z = 0; z < 3; z++) {
					int inner_idx = (z * 9 + y_1 * 3 + x_1) * 3;
					tmp += vectorfield_data[idx + inner_idx + 2];
				//}

				if (tmp < 1.0f / 3.0f * max_acc) {
					dst[row * width * 3 + col * 3 + 0] = (int)(tmp / (1.0f / 3.0f * max_acc) * 254);
				} else if (tmp < 2.0f / 3.0f * max_acc) {
					dst[row * width * 3 + col * 3 + 0] = (int)(254 + ((tmp - 1.0f / 3.0f * max_acc) / (2.0f / 3.0f * max_acc - (1.0f / 3.0f * max_acc))) * (0 - 254));
					dst[row * width * 3 + col * 3 + 1] = (int)((tmp - (1.0f / 3.0f * max_acc)) / ((2.0f / 3.0f * max_acc) - (1.0f / 3.0f * max_acc)) * 254);
				} else {
					dst[row * width * 3 + col * 3 + 1] = (int)(254 + ((tmp - (2.0f / 3.0f * max_acc)) / (max_acc - (2.0f / 3.0f * max_acc))) * (0 - 254));
					dst[row * width * 3 + col * 3 + 2] = (int)(((tmp - (2.0f / 3.0f * max_acc)) / (max_acc - (2.0f / 3.0f * max_acc))) * 254);
				}
			}
		}
	}
}

void statistics_3d_kernel_launch(const float *heatmap_data, const float *vectorfield_data, const float max_vel, const float max_acc, unsigned char* dst, const int width, const int height, struct vector3<int> heatmap_dims, int z_axis) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	statistics_3d_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (heatmap_data, vectorfield_data, max_vel, max_acc, dst, width, height, heatmap_dims, z_axis);
	cudaStreamSynchronize(cuda_streams[3]);
}

__forceinline__
__device__ int statistics_heatmap_get_base_idx(vector3<int> heatmap_dimensions, vector3<float> heatmap_quantization_factors, vector3<int> heatmap_span_start, vector3<float> position) {
	int d_z = (int)floorf((position[2] - heatmap_span_start[2]) * heatmap_quantization_factors[2]);
	if (d_z < 0 || d_z >= heatmap_dimensions[2]) return -1;
	int d_y = (int)floorf((position[1] - heatmap_span_start[1]) * heatmap_quantization_factors[1]);
	if (d_y < 0 || d_y >= heatmap_dimensions[1]) return -1;
	d_y *= heatmap_dimensions[2];
	int d_x = (int)floorf((position[0] - heatmap_span_start[0]) * heatmap_quantization_factors[0]);
	if (d_x < 0 || d_x >= heatmap_dimensions[0]) return -1;
	d_x *= heatmap_dimensions[2] * heatmap_dimensions[1];

	return d_x + d_y + d_z;
}

__global__ void statistics_evolutionary_tracker_single_ray_estimates_kernel(const struct vector3<float>* ray_matrix_device, const int camera_count, const int cdh_max_size, struct vector2<float>* single_ray_position_estimate_device, const int single_ray_max_estimates, const vector3<int> heatmap_dimensions, const vector3<float> heatmap_quantization_factors, const vector3<int> heatmap_span_start, const float* heatmap_device_ptr) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < camera_count * cdh_max_size) {
		int camera_id = i / (cdh_max_size);
		int direction_id = i % cdh_max_size;
		int estimates = 0;
		struct vector3<float> camera_position = ray_matrix_device[camera_id * (1 + cdh_max_size)];
		struct vector3<float> position_tmp = camera_position;
		struct vector3<float> direction = ray_matrix_device[camera_id * (1 + cdh_max_size) + direction_id];
		int idx = statistics_heatmap_get_base_idx(heatmap_dimensions, heatmap_quantization_factors, heatmap_span_start, position_tmp);
		float lambda_min = 10000000.0f;
		float smallest_p = 1.0f;
		int smallest_p_id = -1;
		while (true) {
			if (idx > -1) {
				float p_value = heatmap_device_ptr[idx];
				if (estimates < single_ray_max_estimates) {
					single_ray_position_estimate_device[camera_id * cdh_max_size * single_ray_max_estimates + direction_id * single_ray_max_estimates + estimates] = { p_value, lambda_min };
					if (p_value < smallest_p) {
						smallest_p = p_value;
						smallest_p_id = estimates;
					}
					estimates++;
				} else if (p_value > smallest_p) {
					single_ray_position_estimate_device[camera_id * cdh_max_size * single_ray_max_estimates + direction_id * single_ray_max_estimates + smallest_p_id] = { p_value, lambda_min };
					
					int min_id = -1;
					for (int e = 0; e < single_ray_max_estimates; e++) {
						if (single_ray_position_estimate_device[camera_id * cdh_max_size * single_ray_max_estimates + direction_id * single_ray_max_estimates + e][0] < smallest_p) {
							smallest_p = single_ray_position_estimate_device[camera_id * cdh_max_size * single_ray_max_estimates + direction_id * single_ray_max_estimates + e][0];
							smallest_p_id = e;
						}
					}
				}

				//next cube
				lambda_min = 10000000.0f;
				for (int d = 0; d < 3; d++) {
					if (direction[d] != 0) {
						float lambda = (floorf((position_tmp[d] - heatmap_span_start[d]) * heatmap_quantization_factors[d]) + (1 - 2 * (direction[d] < 0)) - position_tmp[d] - heatmap_span_start[d]) / direction[d];
						if (lambda < lambda_min) lambda_min = lambda;
					}
				}

				position_tmp = position_tmp - -direction * lambda_min;
				int tmp_idx = statistics_heatmap_get_base_idx(heatmap_dimensions, heatmap_quantization_factors, heatmap_span_start, position_tmp);
				if (tmp_idx == idx) {
					position_tmp = position_tmp - -direction * (lambda_min + (heatmap_quantization_factors[0]*0.25f));
					tmp_idx = statistics_heatmap_get_base_idx(heatmap_dimensions, heatmap_quantization_factors, heatmap_span_start, position_tmp);
				}
				idx = tmp_idx;
			} else {
				//blank result
				if (estimates < single_ray_max_estimates) {
					single_ray_position_estimate_device[camera_id * cdh_max_size * single_ray_max_estimates + direction_id * single_ray_max_estimates + + estimates] = { 0.0f, 0.0f };
					break;
				}
			}
		}
	}
}

void statistics_evulotionary_tracker_single_ray_estimates_kernel_launch_async(const struct vector3<float>* ray_matrix_device, const int camera_count, const int cdh_max_size, struct vector2<float>* single_ray_position_estimate_device, const int single_ray_max_estimates, const vector3<int> heatmap_dimensions, const vector3<float> heatmap_quantization_factors, const vector3<int> heatmap_span_start, const float* heatmap_device_ptr, int cuda_stream_index) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (camera_count * cdh_max_size + threadsPerBlock - 1) / threadsPerBlock;
	statistics_evolutionary_tracker_single_ray_estimates_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[cuda_stream_index] >> > (ray_matrix_device, camera_count, cdh_max_size, single_ray_position_estimate_device, single_ray_max_estimates, heatmap_dimensions, heatmap_quantization_factors, heatmap_span_start, heatmap_device_ptr);
}

__global__ void statistics_evolutionary_tracker_kernel(const float *distance_matrix, const struct vector3<float>* min_dist_central_points_matrix_device, const int max_tracked_objects, const unsigned int camera_count, const unsigned int cdh_max_size, int* population, float* scores, const unsigned int population_c, const struct vector2<float>* single_ray_position_estimate_device, const int single_ray_max_estimates, const vector3<int> heatmap_dimensions, const vector3<float> heatmap_quantization_factors, const vector3<int> heatmap_span_start, const float* heatmap_device_ptr) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < population_c) {
		/*
		const bool					*class_match_matrix				= (bool*)memory_pool;
		const float					*distance_matrix				= (float*)(&memory_pool[camera_count * cdh_max_size * camera_count * cdh_max_size * sizeof(bool)]);
		const struct vector3<float> *min_dist_central_points_matrix = (struct vector3<float> *)(&memory_pool[camera_count * cdh_max_size * camera_count * cdh_max_size * sizeof(bool) + camera_count * cdh_max_size * camera_count * cdh_max_size * sizeof(float)]);
		const float					*size_factor_matrix				= (float*)(&memory_pool[camera_count * cdh_max_size * camera_count * cdh_max_size * sizeof(bool) + camera_count * cdh_max_size * camera_count * cdh_max_size * sizeof(float) + camera_count * cdh_max_size * camera_count * cdh_max_size * sizeof(struct vector3<float>)]);
		*/

		//evaluate score
		float score = 0.0f;
		int* pop_object_base_idx = &population[i * (1 + (max_tracked_objects * camera_count))];

		int object_count = pop_object_base_idx[0];
		pop_object_base_idx++;

		int single_rays = 0;
		float avg_score_per_object = 0.0f;
		for (int o = 0; o < object_count; o++) {
			int c_id_last = -1;
			int r_id_last = -1;
			int ray_count = 0;
			float object_score = 0.0f;
			vector3<float> position = { 0.0f, 0.0f, 0.0f };
			for (int c = 0; c < camera_count; c++) {
				int r_id = pop_object_base_idx[0];
				if (r_id > -1) {
					if (c_id_last > -1) {
						float dist_value = distance_matrix[c_id_last * cdh_max_size * camera_count * cdh_max_size + r_id_last * camera_count * cdh_max_size + c * cdh_max_size + r_id];
						position = position - -(min_dist_central_points_matrix_device[c_id_last * cdh_max_size * camera_count * cdh_max_size + r_id_last * camera_count * cdh_max_size + c * cdh_max_size + r_id]);
						if (dist_value > 2.0f) {
							object_score += 100000.0f;
						}
					}
					c_id_last = c;
					r_id_last = r_id;
					ray_count++;
				}
				pop_object_base_idx++;
			}
			if (ray_count == 0) {
				score += 10000000.0f;
			} else if (ray_count == 1) {
				single_rays++;

				float max_p = 0;
				int max_p_id = -1;
				for (int pe = 0; pe < single_ray_max_estimates; pe++) {
					struct vector2<float> est = single_ray_position_estimate_device[c_id_last * cdh_max_size * single_ray_max_estimates + r_id_last * single_ray_max_estimates + pe];
					if (est[0] > max_p) {
						max_p = est[0];
					}
				}

			} else {
				position = position * (1.0f / (float)ray_count);

				int h_idx = statistics_heatmap_get_base_idx(heatmap_dimensions, heatmap_quantization_factors, heatmap_span_start, position);
				if (h_idx > -1) {
					float p_there = heatmap_device_ptr[h_idx];
					object_score *= 1.0f - p_there;
				} else {
					object_score += 1000000.0f;
				}

				avg_score_per_object += object_score;
				score += object_score;
			}
		}
		
		scores[i] = score;
		
		//score penalty for single ray objects
		if (single_rays > 0 && object_count - single_rays > 0) {
			scores[i] += single_rays * (avg_score_per_object/((float)(object_count - single_rays)));
		}
	}
}

void statistics_evolutionary_tracker_kernel_launch(const float* distance_matrix, const struct vector3<float>* min_dist_central_points_matrix_device, const int max_tracked_objects, const unsigned int camera_count, const unsigned int cdh_max_size, int* population, float* scores, const unsigned int population_c, const struct vector2<float>* single_ray_position_estimate_device, const int single_ray_max_estimates, const vector3<int> heatmap_dimensions, const vector3<float> heatmap_quantization_factors, const vector3<int> heatmap_span_start, const float* heatmap_device_ptr) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (population_c + threadsPerBlock - 1) / threadsPerBlock;
	statistics_evolutionary_tracker_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (distance_matrix, min_dist_central_points_matrix_device, max_tracked_objects, camera_count, cdh_max_size, population, scores, population_c, single_ray_position_estimate_device, single_ray_max_estimates, heatmap_dimensions, heatmap_quantization_factors, heatmap_span_start, heatmap_device_ptr);
	cudaStreamSynchronize(cuda_streams[3]);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		logger("CUDA Error etkl: %s\n", cudaGetErrorString(err));
	}
}

__global__ void statistics_evolutionary_tracker_population_evolve_kernel(const int max_tracked_objects, const int camera_count, const int cdh_max_size, int* population, unsigned char* evolution_buffer, float* scores, const unsigned int population_c, const unsigned int population_kept, float mutation_rate, float *randoms, int randoms_size, int min_objects, int max_objects) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < population_c - population_kept) {
		int* pop_object_base = &population[(population_kept + i) * (1 + (max_tracked_objects * camera_count))];
		int object_count = pop_object_base[0];
		pop_object_base++;

		
		//int random_1 = i % randoms_size;
		int random_2 = (i * population_c) % randoms_size;
		//int random_3 = (i * population_kept) % randoms_size;
		//int random_4 = (i * population_kept * population_c) % randoms_size;
		//int random_5 = (i * randoms_size) % randoms_size;
																	  //object selected { i, swap_src }	  //ray selected { i, swap_src }
		unsigned char* evolution_buffer_base = &evolution_buffer[i * (2 * max_tracked_objects			+ 2 * (camera_count * cdh_max_size))];
		
		/*
		//-------------//
		//mutation base//
		//-------------//
		int mutated_objects = randoms[random_1] * (object_count / 4.0f);

		int mutated = 0;
		while (mutated < mutated_objects) {
			int random = (i + mutated * camera_count) % randoms_size;
			int random_r = (i * population_c + mutated) % randoms_size;

			int s_o_id = randoms[random] * object_count;
			if (s_o_id == object_count) s_o_id--;

			int ray_offset = randoms[random_r] * camera_count;
			if (ray_offset == camera_count) ray_offset--;

			int r_o_id = -1;

			for (int r = 0; r < camera_count; r++) {
				int r_o = (r + ray_offset) % camera_count;
				if (pop_object_base[s_o_id * camera_count + r_o] > -1) {
					r_o_id = r_o;
					break;
				}
			}

			int random_s = (random_r + 1) % randoms_size;
			int s_o_id_offset = randoms[random_s] * object_count;

			for (int o = 0; o < object_count; o++) {
				int s_o = (o + s_o_id_offset) % object_count;
				if (s_o != s_o_id) {
					if (pop_object_base[s_o * camera_count + r_o_id] > -1) {
						int ray_id = pop_object_base[s_o * camera_count + r_o_id];
						pop_object_base[s_o * camera_count + r_o_id] = pop_object_base[s_o_id * camera_count + r_o_id];
						pop_object_base[s_o_id * camera_count + r_o_id] = ray_id;
						break;
					}
				}
			}

			mutated++;
		}
		*/

		//----------------------------------------------------//
		//find maximum amount of objects before, mutation fill//
		//----------------------------------------------------//
		int p = (int)(randoms[random_2] * (float)(population_kept - 1));
		int* partner = &population[p * (1 + (max_tracked_objects * camera_count))];
		int partner_object_count = partner[0];
		partner++;
		
		int cross_objs = (int)ceilf((float)(object_count + partner_object_count) * 0.5f);
		if (cross_objs >= max_objects) cross_objs = max_objects;

		int o_1 = 0;
		int o_2 = 0;
		int selected_count = 0;
		int used_rays = 0;
		while (selected_count < cross_objs) {
			while (o_1 < object_count && selected_count < cross_objs) {
				bool blocked_1 = false;
				for (int r = 0; r < camera_count; r++) {
					int ray_id = pop_object_base[o_1 * camera_count + r];
					if (ray_id > -1) {
						if (evolution_buffer_base[(2 * max_tracked_objects) + (r * cdh_max_size) + ray_id]) {
							blocked_1 = true;
							break;
						}
					}
				}
				if (!blocked_1) {
					evolution_buffer_base[2 * o_1] = 1;
					selected_count++;
					for (int r = 0; r < camera_count; r++) {
						int ray_id = pop_object_base[o_1 * camera_count + r];
						if (ray_id > -1) {
							evolution_buffer_base[(2 * max_tracked_objects) + (r * cdh_max_size) + ray_id] = 1;
							used_rays++;
						}
					}
					o_1++;
					break;
				}
				o_1++;
			}
			while (o_2 < partner_object_count && selected_count < cross_objs) {
				bool blocked_2 = false;
				for (int r = 0; r < camera_count; r++) {
					int ray_id = partner[o_2 * camera_count + r];
					if (ray_id > -1) {
						if (evolution_buffer_base[(2 * max_tracked_objects) + (r * cdh_max_size) + ray_id]) {
							blocked_2 = true;
							break;
						}
					}
				}
				if (!blocked_2) {
					evolution_buffer_base[2 * o_1 + 1] = 1;
					selected_count++;
					for (int r = 0; r < camera_count; r++) {
						int ray_id = partner[o_2 * camera_count + r];
						if (ray_id > -1) {
							evolution_buffer_base[(2 * max_tracked_objects) + (r * cdh_max_size) + ray_id] = 1;
							used_rays++;
						}
					}
					o_2++;
					break;
				}
				o_2++;
			}
			if (o_1 >= object_count && o_2 >= partner_object_count) break;
		}
		o_2 = -1;
		int to_add = cross_objs;
		for (int o = 0; o < cross_objs; o++) {
			if (!evolution_buffer_base[2 * o_1]) {
				for (o_2 += 1; o_2 < partner_object_count; o_2++) {
					if (evolution_buffer_base[2 * o_1 + 1]) {
						for (int r = 0; r < camera_count; r++) {
							pop_object_base[o * camera_count + r] = partner[o_2 * camera_count + r];
						}
						to_add--;
						break;
					}
				}
			} else {
				to_add--;
			}
		}
		for (int o = 0; o < partner_object_count; o++) {
			for (int r = 0; r < camera_count; r++) {
				int ray_id = partner[o * camera_count + r];
				if (ray_id > -1 && !evolution_buffer_base[(2 * max_tracked_objects) + (r * cdh_max_size) + ray_id]) {
					for (int o_2 = cross_objs - 1; o_2 >= 0; o_2--) {
						int o_2_o = o_2 - r;
						if (o_2_o < 0) o_2_o += cross_objs;
						if (pop_object_base[o_2_o * camera_count + r] == -1) {
							pop_object_base[o_2_o * camera_count + r] = ray_id;
							used_rays++;
							break;
						}
					}
				}
			}
		}
		for (int o = 0; o < cross_objs; o++) {
			bool empty = true;
			for (int r = 0; r < camera_count; r++) {
				if (pop_object_base[o * camera_count + r] > -1) {
					empty = false;
					break;
				}
			}
			if (empty) {
				if (o < cross_objs - 1) {
					for (int r = 0; r < camera_count; r++) {
						pop_object_base[o * camera_count + r] = pop_object_base[(cross_objs - 1) * camera_count + r];
					}
					cross_objs--;
					o--;
				} else {
					cross_objs--;
				}
			}
		}
		pop_object_base--;
		pop_object_base[0] = cross_objs;

		/*
		//---------------------------------------------------------------------------------//
		//find minimum amount of required object swaps, to have matching participating rays//
		//---------------------------------------------------------------------------------//
		
		//pick random object in i's genetic to swap
		int rand_src_o1 = (int)(randoms[random_1] * (float)(object_count - 1));
		evolution_buffer_base[rand_src_o1 * 2] = 1;

		//mark participating rays on i
		for (int r = 0; r < camera_count; r++) {
			int ray_id_1 = pop_object_base[rand_src_o1 * camera_count + r];
			if (ray_id_1 > -1) {
				evolution_buffer_base[(2 * max_tracked_objects) + (r * 2) * cdh_max_size + (ray_id_1 * 2)] = 1;
			}
		}

		bool pop_object_missing_rays = false;
		bool swap_source_missing_rays = true;

		int pop_obj_swap_count = 1;
		int swap_src_swap_count = 0;

		//pick random partner in population
		int p = (int)(randoms[random_2] * (float)(population_kept - 1));
		int* pop_object_swap_src = &population[p * (1 + (max_tracked_objects * camera_count))];
		int swap_src_object_count = pop_object_swap_src[0];
		pop_object_swap_src++;

		while (pop_object_missing_rays || swap_source_missing_rays) {
			if (swap_source_missing_rays) {
				//add swap source objects, that contain rays of the chosen pop object
				for (int o = 0; o < swap_src_object_count; o++) {
					//if object is not yet selected
					if (!evolution_buffer_base[(2 * o) + 1]) {
						bool take_object = false;
						for (int r = 0; r < camera_count; r++) {
							int ray_id = pop_object_swap_src[o * camera_count + r];
							if (ray_id > -1 && evolution_buffer_base[(2 * max_tracked_objects) + (r * 2) * cdh_max_size + (ray_id * 2)]) {
								take_object = true;
								break;
							}
						}
						if (take_object) {
							swap_src_swap_count++;
							evolution_buffer_base[o * 2 + 1] = 1;
							for (int r = 0; r < camera_count; r++) {
								int ray_id = pop_object_swap_src[o * camera_count + r];
								if (ray_id > -1) {
									evolution_buffer_base[(2 * max_tracked_objects) + (r * 2) * cdh_max_size + (ray_id * 2) + 1] = 1;
									if (!evolution_buffer_base[(2 * max_tracked_objects) + (r * 2) * cdh_max_size + (ray_id * 2)]) {
										pop_object_missing_rays = true;
									}
								}
							}
						}
					}
				}
				swap_source_missing_rays = false;
			}

			if (pop_object_missing_rays) {
				//add pop object objects, that contain rays of the chosen swap source objects
				for (int o = 0; o < object_count; o++) {
					//if object is not yet selected
					if (!evolution_buffer_base[(2 * o)]) {
						bool take_object = false;
						for (int r = 0; r < camera_count; r++) {
							int ray_id = pop_object_base[o * camera_count + r];
							if (ray_id > -1 && evolution_buffer_base[(2 * max_tracked_objects) + (r * 2) * cdh_max_size + (ray_id * 2) + 1]) {
								take_object = true;
								break;
							}
						}
						if (take_object) {
							pop_obj_swap_count++;
							evolution_buffer_base[o * 2] = 1;
							for (int r = 0; r < camera_count; r++) {
								int ray_id = pop_object_base[o * camera_count + r];
								if (ray_id > -1) {
									evolution_buffer_base[(2 * max_tracked_objects) + (r * 2) * cdh_max_size + (ray_id * 2)] = 1;
									if (!evolution_buffer_base[(2 * max_tracked_objects) + (r * 2) * cdh_max_size + (ray_id * 2) + 1]) {
										swap_source_missing_rays = true;
									}
								}
							}
						}
					}
				}
				pop_object_missing_rays = false;
			}
		}

		int final_object_count = object_count + (swap_src_swap_count - pop_obj_swap_count);
		
		if (final_object_count >= min_objects && final_object_count <= max_objects && swap_src_swap_count < swap_src_object_count) {
			//perform swap
			int swap_source_swapped = 0;
			int swap_source_last_idx = -1;
			bool erased_something = false;
			for (int o = 0; o < object_count; o++) {
				if (evolution_buffer_base[o * 2]) {
					if (swap_source_swapped < swap_src_swap_count) {
						//override
						for (int ssli = swap_source_last_idx + 1; ssli < swap_src_object_count; ssli++) {
							if (evolution_buffer_base[ssli * 2 + 1]) {
								swap_source_last_idx = ssli;
								for (int r = 0; r < camera_count; r++) {
									int ray_id = pop_object_swap_src[o * camera_count + r];
									pop_object_base[o * camera_count + r] = ray_id;
								}
								swap_source_swapped++;
								break;
							}
						}
					} else {
						//erase object
						erased_something = true;
						for (int r = 0; r < camera_count; r++) {
							pop_object_base[o * camera_count + r] = -1;
						}
					}
				}
			}
			//add remaining objects
			if (swap_source_swapped < swap_src_swap_count) {
				for (int o = object_count; o < max_tracked_objects; o++) {
					for (int ssli = swap_source_last_idx + 1; ssli < swap_src_object_count; ssli++) {
						if (evolution_buffer_base[ssli * 2 + 1]) {
							swap_source_last_idx = ssli;
							for (int r = 0; r < camera_count; r++) {
								int ray_id = pop_object_swap_src[o * camera_count + r];
								pop_object_base[o * camera_count + r] = ray_id;
							}
							swap_source_swapped++;
							break;
						}
					}
					if (swap_source_swapped == swap_src_swap_count) break;
				}
			}
			if (erased_something) {
				for (int o = 0; o < final_object_count; o++) {
					bool empty = true;
					for (int r = 0; r < camera_count; r++) {
						if (pop_object_base[o * camera_count + r] > -1) {
							empty = false;
							break;
						}
					}
					if (empty) {
						if (o < object_count - 1) {
							for (int r = 0; r < camera_count; r++) {
								pop_object_base[o * camera_count + r] = pop_object_base[(object_count - 1) * camera_count + r];
							}
							object_count--;
							o--;
						}
					}
				}
			}
			pop_object_base--;
			pop_object_base[0] = final_object_count;
		} else {
			if (randoms[random_3] < mutation_rate) {
				int c_id = (int)(randoms[random_4] * camera_count);
				if (c_id == camera_count) c_id--;
				int o_offset = (int)randoms[random_5] * object_count;

				int o_base_id = -1;
				for (int o = 0; o < object_count; o++) {
					int o_o = (o + o_offset) % object_count;
					if (o_base_id == -1) {
						if (pop_object_base[o_o * camera_count + c_id] > -1) {
							o_base_id = o_o;
						}
					} else {
						if (pop_object_base[o_o * camera_count + c_id] > -1) {
							int ray_id = pop_object_base[o_base_id * camera_count + c_id];
							pop_object_base[o_base_id * camera_count + c_id] = pop_object_base[o_o * camera_count + c_id];
							pop_object_base[o_o * camera_count + c_id] = ray_id;
						}
					}
				}
			}
		}
		*/
	}
}

void statistics_evolutionary_tracker_population_evolve_kernel_launch(const int max_tracked_objects, const int camera_count, const int cdh_max_size, int* population, unsigned char* evolution_buffer, float* scores, const unsigned int population_c, const float population_keep_factor, float mutation_rate, float* randoms, int randoms_size, int min_objects, int max_objects) {
	int threadsPerBlock = 256;
	int population_kept = (int)floorf(population_c * population_keep_factor);

	int blocksPerGrid = (population_c - population_kept + threadsPerBlock - 1) / threadsPerBlock;
	statistics_evolutionary_tracker_population_evolve_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (max_tracked_objects, camera_count, cdh_max_size, population, evolution_buffer, scores, population_c, population_kept, mutation_rate, randoms, randoms_size, min_objects, max_objects);
	cudaStreamSynchronize(cuda_streams[3]);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		logger("CUDA Error etpekl: %s\n", cudaGetErrorString(err));
	}
}
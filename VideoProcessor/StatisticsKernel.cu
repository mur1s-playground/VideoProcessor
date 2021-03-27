#include "StatisticsKernel.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

#include "Vector2.h"

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


__global__ void statistics_3d_kernel(const float* heatmap_data, const float* vectorfield_data, const float max_vel, const float max_acc, unsigned char* dst, const int width, const int height, struct vector3<int> heatmap_dims) {
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

		world_size_x = { 0.0f, 10.0f * 20.0f };
		world_size_y = { 0.0f, 25.0f * 20.0f };

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

				int idx = position[0] * heatmap_dims[1] * heatmap_dims[2] + position[1] * heatmap_dims[2] + 0;

				float tmp = 0.0f;
				for (int z = 0; z < heatmap_dims[2]; z++) {
					tmp += heatmap_data[idx + z];
				}
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

				int idx = position[0] * heatmap_dims[1] * heatmap_dims[2] + position[1] * heatmap_dims[2] + 1;
				idx *= 27 * 3;

				dst[row * width * 3 + col * 3 + 0] = 0;
				dst[row * width * 3 + col * 3 + 1] = 0;
				dst[row * width * 3 + col * 3 + 2] = 0;

				float tmp = 0.0f;

				int z = 1;
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

				int idx = position[0] * heatmap_dims[1] * heatmap_dims[2] + position[1] * heatmap_dims[2] + 1;
				idx *= 27 * 3;

				dst[row * width * 3 + col * 3 + 0] = 0;
				dst[row * width * 3 + col * 3 + 1] = 0;
				dst[row * width * 3 + col * 3 + 2] = 0;

				float tmp = 0.0f;

				int z = 1;
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

				int idx = position[0] * heatmap_dims[1] * heatmap_dims[2] + position[1] * heatmap_dims[2] + 1;
				idx *= 27 * 3;

				dst[row * width * 3 + col * 3 + 0] = 0;
				dst[row * width * 3 + col * 3 + 1] = 0;
				dst[row * width * 3 + col * 3 + 2] = 0;

				float tmp = 0.0f;

				int z = 1;
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

void statistics_3d_kernel_launch(const float *heatmap_data, const float *vectorfield_data, const float max_vel, const float max_acc, unsigned char* dst, const int width, const int height, struct vector3<int> heatmap_dims) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	statistics_3d_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (heatmap_data, vectorfield_data, max_vel, max_acc, dst, width, height, heatmap_dims);
	cudaStreamSynchronize(cuda_streams[3]);
}
#include "CameraControlDiagnostic.h"
#include "CameraControlDiagnosticKernel.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

__global__ void camera_control_diagnostic_kernel(const unsigned char* gpu_shared_state, const unsigned int camera_count, unsigned char* dst, const int width, const int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / width;
		int col = (i % width);

		int screen_height = (height / 3);
		int screen = row / screen_height;
		float screen_ratio = width / (float)screen_height;

		dst[row * width * 3 + col * 3 + 0] = 0;
		dst[row * width * 3 + col * 3 + 1] = 0;
		dst[row * width * 3 + col * 3 + 2] = 0;

		struct camera_control_shared_state* ccss = (struct camera_control_shared_state *)gpu_shared_state;

		struct vector2<float> world_size_x = { 100000.0f, -100000.0f };
		struct vector2<float> world_size_y = { 100000.0f, -100000.0f };
		struct vector2<float> world_size_z = { 100000.0f, -100000.0f };

		float scaling_factor = 20.0f;

		for (int c = 0; c < camera_count; c++) {
			if (ccss[c].position[0] * scaling_factor < world_size_x[0]) {
				world_size_x[0] = ccss[c].position[0]* scaling_factor;
			}
			if (ccss[c].position[0] * scaling_factor > world_size_x[1]) {
				world_size_x[1] = ccss[c].position[0]* scaling_factor;
			}

			if (ccss[c].position[1] * scaling_factor < world_size_y[0]) {
				world_size_y[0] = ccss[c].position[1]* scaling_factor;
			}
			if (ccss[c].position[1] * scaling_factor > world_size_y[1]) {
				world_size_y[1] = ccss[c].position[1]* scaling_factor;
			}

			if (ccss[c].position[2] * scaling_factor < world_size_z[0]) {
				world_size_z[0] = ccss[c].position[2]* scaling_factor;
			}
			if (ccss[c].position[2] * scaling_factor > world_size_z[1]) {
				world_size_z[1] = ccss[c].position[2]* scaling_factor;
			}
		}

		world_size_x[0] -= (0.2f * max(100.0f, (world_size_x[1] - world_size_x[0])));
		world_size_x[1] += (0.2f * max(100.0f, (world_size_x[1] - world_size_x[0])));

		world_size_y[0] -= (0.2f * max(100.0f, (world_size_y[1] - world_size_y[0])));
		world_size_y[1] += (0.2f * max(100.0f, (world_size_y[1] - world_size_y[0])));

		world_size_z[0] -= (0.2f * max(100.0f, (world_size_z[1] - world_size_z[0])));
		world_size_z[1] += (0.2f * max(100.0f, (world_size_z[1] - world_size_z[0])));

		struct vector3<float> world_dim = { world_size_x[1] - world_size_x[0], world_size_y[1] - world_size_y[0], world_size_z[1] - world_size_z[0]};

		float world_ratio_xy = world_dim[0] / world_dim[1];
		if (world_ratio_xy > screen_ratio) {
			world_size_y[0] = world_size_y[0] / world_ratio_xy * screen_ratio;
			world_size_y[1] = world_size_y[1] / world_ratio_xy * screen_ratio;
		} else {
			float world_dim_x_diff = (screen_ratio*world_dim[1]) - world_dim[0];

			world_size_x[0] = world_size_x[0] - (world_dim_x_diff / 2.0f);
			world_size_x[1] = world_size_x[1] + (world_dim_x_diff / 2.0f);
		}

		world_dim = { world_size_x[1] - world_size_x[0], world_size_y[1] - world_size_y[0], world_size_z[1] - world_size_z[0] };

		float pixel_width = world_dim[0] / (float)width;

		if (screen == 0) {
			struct vector2<float>	world_position = {
				world_size_x[0] + (col / (float)width) * (world_size_x[1] - world_size_x[0]),
				world_size_y[0] + (((screen_height-row) % screen_height) / (float)(screen_height)) * (world_size_y[1] - world_size_y[0])
			};

			/*//draw grid
			if (world_position[0] - floor(world_position[0]) < pixel_width) {
				dst[row * width * 3 + col * 3 + 0] = 0;
				dst[row * width * 3 + col * 3 + 1] = 255;
				dst[row * width * 3 + col * 3 + 2] = 0;
			}
			if (world_position[1] - floor(world_position[1]) < pixel_width) {
				dst[row * width * 3 + col * 3 + 0] = 255;
				dst[row * width * 3 + col * 3 + 1] = 0;
				dst[row * width * 3 + col * 3 + 2] = 0;
			}
			*/

			//draw cc
			{
				float fade_width = 5.0f;
				float len_diff = (10.0f * pixel_width) - length(world_position);
				if (len_diff > fade_width * pixel_width) {
					dst[row * width * 3 + col * 3 + 0] = 128;
					dst[row * width * 3 + col * 3 + 1] = 0;
					dst[row * width * 3 + col * 3 + 2] = 128;
				} else if (len_diff >= 0) {
					dst[row * width * 3 + col * 3 + 0] = 128 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 0] - 128);
					dst[row * width * 3 + col * 3 + 1] = 0 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 1] - 0);
					dst[row * width * 3 + col * 3 + 2] = 128 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 2] - 128);
				}
			}

			//draw fov
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				cam_2d_pos = cam_2d_pos * scaling_factor;

				float fade_width = 5.0f;
				float len = length(cam_2d_pos - world_position);
				float len_diff = (35.0f * pixel_width) - len;

				if (len > 10.0f * pixel_width && len_diff >= 0.0f) {
					struct vector2<float> pos_diff = cam_2d_pos - world_position;
					float phi = 0.0f;
					if (pos_diff[1] >= 0) {
						phi = acos(pos_diff[0] / len);
					} else {
						phi = -acos(pos_diff[0] / len);
					}
					phi *= (2.0f * 90.0f) / 3.14159265358979323846f;
					phi += ccss[c].np_angle + 90.0f;
					while (phi > 180.0f) phi -= 360.0f;

					struct vector2<float> fov = ccss[c].fov;
					if (fov[0] == 0.0f) fov[0] = 40.0f;
					float p_b = -(fov[0] / 2.0f);
					float b_e = (fov[0] / 2.0f);
					if (phi >= p_b && phi <= b_e) {
						if (phi < -(fov[0] / 2.0f - fade_width * 2.0f)) {
							float fade_l = -(fade_width * pixel_width - len_diff < 0) + (fade_width * pixel_width - len_diff > 0) * ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width));
							float fade = (-(fov[0] / 2.0f - fade_width * 2.0f) - phi) / (fade_width * 2.0f);
							if (fade_l > fade) {
								fade = fade_l;
							}
							dst[row * width * 3 + col * 3 + 0] = 128 + fade * (dst[row * width * 3 + col * 3 + 0] - 128);
							dst[row * width * 3 + col * 3 + 1] = 0 + fade * (dst[row * width * 3 + col * 3 + 1] - 0);
							dst[row * width * 3 + col * 3 + 2] = 128 + fade * (dst[row * width * 3 + col * 3 + 2] - 128);
						} else if (phi > (fov[0] / 2.0f - fade_width * 2.0f)) {
							float fade_l = -(fade_width * pixel_width - len_diff < 0) + (fade_width * pixel_width - len_diff > 0) * ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width));
							float fade = (phi - (fov[0] / 2.0f - fade_width * 2.0f)) / (fade_width * 2.0f);
							if (fade_l > fade) {
								fade = fade_l;
							}
							dst[row * width * 3 + col * 3 + 0] = 128 + fade * (dst[row * width * 3 + col * 3 + 0] - 128);
							dst[row * width * 3 + col * 3 + 1] = 0 + fade * (dst[row * width * 3 + col * 3 + 1] - 0);
							dst[row * width * 3 + col * 3 + 2] = 128 + fade * (dst[row * width * 3 + col * 3 + 2] - 128);
						} else {
							if (len_diff > fade_width * pixel_width) {
								dst[row * width * 3 + col * 3 + 0] = 128;
								dst[row * width * 3 + col * 3 + 1] = 0;
								dst[row * width * 3 + col * 3 + 2] = 128;
							} else {
								dst[row * width * 3 + col * 3 + 0] = 128 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 0] - 128);
								dst[row * width * 3 + col * 3 + 1] = 0 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 1] - 0);
								dst[row * width * 3 + col * 3 + 2] = 128 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 2] - 128);
							}
						}
					}
				}
			}


			//draw camera
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				cam_2d_pos = cam_2d_pos * scaling_factor;

				float fade_width = 5.0f;
				float len_diff = (17.5f * pixel_width) - length(cam_2d_pos - world_position);
				
				if (len_diff > fade_width*pixel_width) {
					dst[row * width * 3 + col * 3 + 0] = 255;
					dst[row * width * 3 + col * 3 + 1] = 255;
					dst[row * width * 3 + col * 3 + 2] = 255;
				} else if (len_diff >= 0) {
					dst[row * width * 3 + col * 3 + 0] = 255 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 0] - 255);
					dst[row * width * 3 + col * 3 + 1] = 255 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 1] - 255);
					dst[row * width * 3 + col * 3 + 2] = 255 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 2] - 255);
				}
			}

			//draw camera_statistic_looking_direction
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				cam_2d_pos = cam_2d_pos * scaling_factor;

				float d_length = 25.0f * pixel_width;
				struct vector2<float> np_dir = { cosf((ccss[c].np_angle - 90.0f) * 3.1415f / (2.0f * 90.0f)), -sinf((ccss[c].np_angle - 90.0f) * 3.1415f / (2.0f * 90.0f)) };

				float min_dist = length(cam_2d_pos - world_position);;

				if (min_dist < d_length) {
					for (int ss = 0; ss < 20; ss++) {
						struct vector2<float> draw_pos = cam_2d_pos - np_dir * (-ss / 20.0f) * d_length;
						if (length(draw_pos - world_position) < min_dist) {
							min_dist = length(draw_pos - world_position);
						}
					}

					if (min_dist < 1.0f * pixel_width) {
						dst[row * width * 3 + col * 3 + 0] = 0;
						dst[row * width * 3 + col * 3 + 1] = 255;
						dst[row * width * 3 + col * 3 + 2] = 0;
					} else if (min_dist < 4.0f * pixel_width) {
						dst[row * width * 3 + col * 3 + 0] = 0 + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 0] - 0.0f));
						dst[row * width * 3 + col * 3 + 1] = 255 + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 1] - 255.0f));
						dst[row * width * 3 + col * 3 + 2] = 0 + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 2] - 0.0f));
					}
				}

			}

			//draw camera_sensor_looking_direction
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				cam_2d_pos = cam_2d_pos * scaling_factor;

				float d_length = 12.5f * pixel_width;
				struct vector2<float> np_dir = { cosf((ccss[c].np_sensor-90.0f) * 3.1415f / (2.0f*90.0f)), -sinf((ccss[c].np_sensor-90.0f) * 3.1415f / (2.0f * 90.0f)) };

				float min_dist = length(cam_2d_pos - world_position);;

				if (min_dist < d_length) {
					for (int ss = 0; ss < 20; ss++) {
						struct vector2<float> draw_pos = cam_2d_pos - np_dir * (-ss / 20.0f) * d_length;
						if (length(draw_pos - world_position) < min_dist) {
							min_dist = length(draw_pos - world_position);
						}
					}

					if (min_dist < 1.0f * pixel_width) {
						dst[row * width * 3 + col * 3 + 0] = 0;
						dst[row * width * 3 + col * 3 + 1] = 0;
						dst[row * width * 3 + col * 3 + 2] = 255;
					} else if (min_dist < 4.0f * pixel_width) {
						dst[row * width * 3 + col * 3 + 0] = 0 + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 0] - 0.0f));
						dst[row * width * 3 + col * 3 + 1] = 0 + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 1] - 0.0f));
						dst[row * width * 3 + col * 3 + 2] = 255 + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 2] - 255.0f));
					}
				}
			}
		}
	}
}


void camera_control_diagnostic_launch(const unsigned char* gpu_shared_state, const unsigned int camera_count, unsigned char* dst, const int width, const int height) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	camera_control_diagnostic_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (gpu_shared_state, camera_count, dst, width, height);
	cudaStreamSynchronize(cuda_streams[3]);
}
#include "CameraControlDiagnostic.h"
#include "CameraControlDiagnosticKernel.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

__forceinline__
__device__ void camera_control_diagnostic_draw_faded_circle_object(struct vector2<float> object_position, float scaling_factor, float fade_radius, float total_radius, struct vector3<unsigned char> color, float pixel_width, struct vector2<float> world_position, unsigned char* dst, int row, int col, int width) {
	object_position = object_position * scaling_factor;
	float fade_width = fade_radius;
	float len_diff = (total_radius * pixel_width) - length(object_position - world_position);
	if (len_diff > fade_width * pixel_width) {
		dst[row * width * 3 + col * 3 + 0] = color[0];
		dst[row * width * 3 + col * 3 + 1] = color[1];
		dst[row * width * 3 + col * 3 + 2] = color[2];
	} else if (len_diff >= 0) {
		dst[row * width * 3 + col * 3 + 0] = color[0] + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 0] - color[0]);
		dst[row * width * 3 + col * 3 + 1] = color[1] + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 1] - color[1]);
		dst[row * width * 3 + col * 3 + 2] = color[2] + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 2] - color[2]);
	}
}

__forceinline__
__device__ void camera_control_diagnostic_draw_looking_direction(struct vector2<float> cam_2d_pos, float scaling_factor, struct vector3<unsigned char> color, float pixel_width, struct vector2<float> world_position, float d_len, unsigned char* dst, int row, int col, int width, float angle) {
	cam_2d_pos = cam_2d_pos * scaling_factor;

	float d_length = d_len * pixel_width;
	struct vector2<float> np_dir = { cosf((angle - 90.0f) * 3.1415f / (2.0f * 90.0f)), -sinf((angle - 90.0f) * 3.1415f / (2.0f * 90.0f)) };

	float min_dist = length(cam_2d_pos - world_position);;

	if (min_dist < d_length) {
		for (int ss = 0; ss < 20; ss++) {
			struct vector2<float> draw_pos = cam_2d_pos - np_dir * (-ss / 20.0f) * d_length;
			if (length(draw_pos - world_position) < min_dist) {
				min_dist = length(draw_pos - world_position);
			}
		}

		if (min_dist < 1.0f * pixel_width) {
			dst[row * width * 3 + col * 3 + 0] = color[0];
			dst[row * width * 3 + col * 3 + 1] = color[1];
			dst[row * width * 3 + col * 3 + 2] = color[2];
		} else if (min_dist < 4.0f * pixel_width) {
			dst[row * width * 3 + col * 3 + 0] = color[0] + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 0] - color[0]));
			dst[row * width * 3 + col * 3 + 1] = color[1] + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 1] - color[1]));
			dst[row * width * 3 + col * 3 + 2] = color[2] + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 2] - color[2]));
		}
	}
}

__forceinline__
__device__ void camera_control_diagnostic_draw_looking_direction_vec(struct vector2<float> cam_2d_pos, float scaling_factor, struct vector3<unsigned char> color, float pixel_width, struct vector2<float> world_position, float d_len, unsigned char* dst, int row, int col, int width, struct vector2<float> direction) {
	cam_2d_pos = cam_2d_pos * scaling_factor;

	float d_length = d_len * pixel_width;
	struct vector2<float> np_dir = direction;

	float min_dist = length(cam_2d_pos - world_position);

	if (min_dist < d_length) {
		for (int ss = 0; ss < 20; ss++) {
			struct vector2<float> draw_pos = cam_2d_pos - np_dir * (-ss / 20.0f) * d_length;
			if (length(draw_pos - world_position) < min_dist) {
				min_dist = length(draw_pos - world_position);
			}
		}

		if (min_dist < 1.0f * pixel_width) {
			dst[row * width * 3 + col * 3 + 0] = color[0];
			dst[row * width * 3 + col * 3 + 1] = color[1];
			dst[row * width * 3 + col * 3 + 2] = color[2];
		}
		else if (min_dist < 4.0f * pixel_width) {
			dst[row * width * 3 + col * 3 + 0] = color[0] + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 0] - color[0]));
			dst[row * width * 3 + col * 3 + 1] = color[1] + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 1] - color[1]));
			dst[row * width * 3 + col * 3 + 2] = color[2] + (min_dist / (4.0f * pixel_width) * (dst[row * width * 3 + col * 3 + 2] - color[2]));
		}
	}
}

__forceinline__
__device__ void camera_control_diagnostic_draw_fov(struct vector2<float> cam_2d_pos, float scaling_factor, float pixel_width, struct vector2<float> world_position, unsigned char* dst, int row, int col, int width, float angle, struct vector2<float> fov) {
	cam_2d_pos = cam_2d_pos * scaling_factor;

	float fade_width = 10.0f;
	float len = length(cam_2d_pos - world_position);
	float len_diff = (50.0f * pixel_width) - len;

	if (len > 10.0f * pixel_width && len_diff >= 0.0f) {
		struct vector2<float> pos_diff = cam_2d_pos - world_position;
		float phi = 0.0f;
		if (pos_diff[1] >= 0) {
			phi = acos(pos_diff[0] / len);
		}
		else {
			phi = -acos(pos_diff[0] / len);
		}
		phi *= (2.0f * 90.0f) / 3.14159265358979323846f;
		phi += angle + 90.0f;
		while (phi > 180.0f) phi -= 360.0f;

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
				}
				else {
					dst[row * width * 3 + col * 3 + 0] = 128 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 0] - 128);
					dst[row * width * 3 + col * 3 + 1] = 0 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 1] - 0);
					dst[row * width * 3 + col * 3 + 2] = 128 + ((fade_width * pixel_width - len_diff) / (fade_width * pixel_width)) * (dst[row * width * 3 + col * 3 + 2] - 128);
				}
			}
		}
	}
}

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

		if (abs(row - screen*screen_height) < 2 * pixel_width) {
			dst[row * width * 3 + col * 3 + 0] = 255;
			dst[row * width * 3 + col * 3 + 1] = 255;
			dst[row * width * 3 + col * 3 + 2] = 255;
		}

		if (screen == 0) {
			struct vector2<float>	world_position = {
				world_size_x[0] + (col / (float)width) * (world_size_x[1] - world_size_x[0]),
				world_size_y[0] + (((screen_height-row) % screen_height) / (float)(screen_height)) * (world_size_y[1] - world_size_y[0])
			};

			//draw cc
			camera_control_diagnostic_draw_faded_circle_object(struct vector2<float>(0.0f, 0.0f), scaling_factor, 5.0f, 10.0f, struct vector3<unsigned char>(128, 0, 128), pixel_width, world_position, dst, row, col, width);

			//draw fov
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				camera_control_diagnostic_draw_fov(cam_2d_pos, scaling_factor, pixel_width, world_position, dst, row, col, width, ccss[c].np_angle, ccss[c].fov);
			}

			//draw detection rays
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				for (int r = 0; r < 5; r++) {
					if (ccss[c].latest_detections_rays[r][0] != 0) {
						camera_control_diagnostic_draw_looking_direction(cam_2d_pos, scaling_factor, struct vector3<unsigned char>(123 + (c * 5 + r) * 13, 123 + (c * 5 + r) * 37, 0), pixel_width, world_position, 150, dst, row, col, width, ccss[c].latest_detections_rays[r][0] - 90.0f);
					}
				}
			}

			//draw detected objects
			for (int c = 0; c < camera_count; c++) {
				for (int r = 0; r < 5; r++) {
					if (ccss[c].latest_detections_objects[r][0] != 0) {
						struct vector2<float> cam_2d_pos = { ccss[c].latest_detections_objects[r][0], ccss[c].latest_detections_objects[r][1] };
						camera_control_diagnostic_draw_faded_circle_object(cam_2d_pos, scaling_factor, 5.0f, 10.0f, struct vector3<unsigned char>(0, 123 + (c * 5 + r) * 13, 123 + (c * 5 + r) * 13), pixel_width, world_position, dst, row, col, width);
					}
				}
			}

			//draw camera
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				camera_control_diagnostic_draw_faded_circle_object(cam_2d_pos, scaling_factor, 5.0f, 17.5f, struct vector3<unsigned char>(255, 255, 255), pixel_width, world_position, dst, row, col, width);
			}

			//draw camera_statistic_looking_direction
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				float d_len = 25.0f;
				camera_control_diagnostic_draw_looking_direction(cam_2d_pos, scaling_factor, struct vector3<unsigned char>(0, 255, 0), pixel_width, world_position, d_len, dst, row, col, width, ccss[c].np_angle);
			}

			//draw camera_sensor_looking_direction
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[1] };
				float d_len = 12.5f;
				camera_control_diagnostic_draw_looking_direction(cam_2d_pos, scaling_factor, struct vector3<unsigned char>(0, 0, 255), pixel_width, world_position, d_len, dst, row, col, width, ccss[c].np_sensor);
			}
		} else if (screen == 1) {
			struct vector2<float>	world_position = {
				world_size_x[0] + (col / (float)width) * (world_size_x[1] - world_size_x[0]),
				world_size_z[0] + ((((screen + 1) * screen_height - row) % screen_height) / (float)(screen_height)) * (world_size_z[1] - world_size_z[0])
			};

			//draw cc
			camera_control_diagnostic_draw_faded_circle_object(struct vector2<float>(0.0f, 0.0f), scaling_factor, 5.0f, 10.0f, struct vector3<unsigned char>(128, 0, 128), pixel_width, world_position, dst, row, col, width);
			
			/*
			//draw fov
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[2] };
				camera_control_diagnostic_draw_fov(cam_2d_pos, scaling_factor, pixel_width, world_position, dst, row, col, width, ccss[c].np_angle, ccss[c].fov);
			}
			*/

			//draw detection rays
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[2] };
				for (int r = 0; r < 5; r++) {
					if (ccss[c].latest_detections_rays[r][1] != 0) {
						float north_pole = ccss[c].latest_detections_rays[r][0];
						float horizon = ccss[c].latest_detections_rays[r][1];

						float M_PI = 3.14159274101257324219;

						struct vector2<float> unit_vec = {						
							-sinf(horizon * (M_PI / (2.0f * 90.0f))) * cosf(north_pole * (M_PI / (2.0f * 90.0f))),
							cosf(horizon * (M_PI / (2.0f * 90.0f)))
						};

						camera_control_diagnostic_draw_looking_direction_vec(cam_2d_pos, scaling_factor, struct vector3<unsigned char>(123 + (c * 5 + r) * 13, 123 + (c * 5 + r) * 37, 0), pixel_width, world_position, 150, dst, row, col, width, unit_vec);
					}
				}
			}

			//draw detected objects
			for (int c = 0; c < camera_count; c++) {
				for (int r = 0; r < 5; r++) {
					if (ccss[c].latest_detections_objects[r][0] != 0) {
						struct vector2<float> cam_2d_pos = { ccss[c].latest_detections_objects[r][0], -ccss[c].latest_detections_objects[r][2] };
						camera_control_diagnostic_draw_faded_circle_object(cam_2d_pos, scaling_factor, 5.0f, 10.0f, struct vector3<unsigned char>(0, 123 + (c * 5 + r) * 13, 123 + (c * 5 + r) * 13), pixel_width, world_position, dst, row, col, width);
					}
				}
			}

			//draw camera
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[2] };
				camera_control_diagnostic_draw_faded_circle_object(cam_2d_pos, scaling_factor, 5.0f, 17.5f, struct vector3<unsigned char>(255, 255, 255), pixel_width, world_position, dst, row, col, width);
			}

			/*
			//draw camera_statistic_looking_direction
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[2] };
				float d_len = 25.0f;
				ccss[c].np_angle

				camera_control_diagnostic_draw_looking_direction(cam_2d_pos, scaling_factor, struct vector3<unsigned char>(0, 255, 0), pixel_width, world_position, d_len, dst, row, col, width, ccss[c].horizon_angle);
			}

			//draw camera_sensor_looking_direction
			for (int c = 0; c < camera_count; c++) {
				struct vector2<float> cam_2d_pos = { ccss[c].position[0], ccss[c].position[2] };
				float d_len = 12.5f;
				camera_control_diagnostic_draw_looking_direction(cam_2d_pos, scaling_factor, struct vector3<unsigned char>(0, 0, 255), pixel_width, world_position, d_len, dst, row, col, width, ccss[c].horizon_sensor);
			}
			*/
		}
	}
}


void camera_control_diagnostic_launch(const unsigned char* gpu_shared_state, const unsigned int camera_count, unsigned char* dst, const int width, const int height) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	camera_control_diagnostic_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (gpu_shared_state, camera_count, dst, width, height);
	cudaStreamSynchronize(cuda_streams[3]);
}
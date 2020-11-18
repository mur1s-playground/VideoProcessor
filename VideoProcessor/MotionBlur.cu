#include "MotionBlur.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

__global__ void motion_blur_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int frame_count, const int frame_id, const int weight_dist_type, const float frame_id_weight_center, const float a, const float b, const float c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * channels) {
		float weight = 0.0f;
		if (weight_dist_type <= 1) { //even or linear hat
			weight = c + a * (frame_id - frame_id_weight_center) * (frame_id < frame_id_weight_center) + b * (frame_id - frame_id_weight_center) * (frame_id > frame_id_weight_center);
		} else if (weight_dist_type == 2) { //other...

		}
		float value = (float)src[i] * weight;

		if (frame_id == 0) {
			if (value > 255.0f) value = 255.0f;
			dst[i] = (unsigned char)value;
		} else {
			value += (float)dst[i];
			if (value > 255.0f) value = 255.0f;
			dst[i] = (unsigned char)value;
		}
	}
}

void motion_blur_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int frame_count, const int frame_id, const int weight_dist_type, const float frame_id_weight_center, const float a, const float b, const float c, bool sync) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * channels + threadsPerBlock - 1) / threadsPerBlock;
	motion_blur_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, frame_count, frame_id, weight_dist_type, frame_id_weight_center, a, b, c);
	if (sync) {
		cudaStreamSynchronize(cuda_streams[3]);
	}
}
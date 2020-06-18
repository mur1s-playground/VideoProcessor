#include "MotionBlur.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

__global__ void motion_blur_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int frame_count, const int frame_id) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * channels) {
		float value = (float)src[i] / (float)frame_count;
				
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

void motion_blur_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int frame_count, const int frame_id, bool sync) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * channels + threadsPerBlock - 1) / threadsPerBlock;
	motion_blur_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, frame_count, frame_id);
	if (sync) {
		cudaStreamSynchronize(cuda_streams[3]);
	}
}
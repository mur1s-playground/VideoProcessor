#include "GreenScreenKernel.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

__global__ void green_screen_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const vector3<unsigned char> rgb, const float threshold) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * channels) {
		int row = i / (channels * width);
		int col = (i % (channels * width)) / channels;
		int channel = (i % (channels * width)) % channels;

		float value = 0.0f;
		int idx = row * (3 * width) + col * 3;

		if (channel < 3) {
			dst[i] = src[idx + channel];
		} else {
			for (int c = 0; c < 3; c++) {
				value += ((float)src[idx + c] - (float)rgb[2 - c])*((float)src[idx + c] - (float)rgb[2 - c]);
			}
			value = sqrtf(value);
			if (value < threshold) {
				dst[i] = 0;
			} else {
				dst[i] = 255;
			}
		}
	}
}


void green_screen_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const vector3<unsigned char> rgb, const float threshold) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * channels + threadsPerBlock - 1) / threadsPerBlock;
	green_screen_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, rgb, threshold);
	cudaStreamSynchronize(cuda_streams[3]);
}
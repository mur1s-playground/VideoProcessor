#include "GaussianBlur.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

#include "math.h"

#include "Logger.h"

void gaussian_blur_construct_kernel(float** host_kernel_out, float* norm_out, const int kernel_size, const float a, const float b, const float c) {
	float* host_kernel = new float[kernel_size * kernel_size];
	float norm = 0;
	for (int x = -kernel_size/2; x <= kernel_size/2; x++) {
		for (int y = -kernel_size/2; y < kernel_size/2; y++) {
			float v = abs(y)+abs(x);
			host_kernel[(y+(kernel_size/2)) * kernel_size + x+(kernel_size/2)] = (a * exp(-((v - b) * (v - b)) / (2 * c * c)));
			norm += host_kernel[(y + (kernel_size / 2)) * kernel_size + x + (kernel_size / 2)];
		}
	}
	*host_kernel_out = host_kernel;
	*norm_out = norm;
}

__global__ void gaussian_blur_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int kernel_size, const float *kernel, const float kernel_norm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * channels) {
		int row = i / (channels * width);
		int col = (i % (channels * width)) / channels;
		int channel = (i % (channels * width)) % channels;

		float value = 0.0f;
		
		for (int y = -kernel_size / 2; y <= kernel_size / 2; y++) {
			for (int x = -kernel_size / 2; x <= kernel_size / 2; x++) {
				if (row + y >= 0 && row + y < height && col + x >= 0 && col + x < width) {
					float base_val = ((float)src[(row + y) * width * channels + (col + x) * channels + channel]);
					value += ((kernel[(y + (kernel_size / 2)) * kernel_size + x + (kernel_size / 2)]/kernel_norm) * base_val);
				}
			}
		}
		if (value > 255.0f) value = 255.0f;
		
		dst[i] = (unsigned char)value;	
	}
}

void gaussian_blur_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int kernel_size, const float *kernel, const float kernel_norm) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * channels + threadsPerBlock - 1) / threadsPerBlock;
	gaussian_blur_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, kernel_size, kernel, kernel_norm);
	cudaStreamSynchronize(cuda_streams[3]);
}
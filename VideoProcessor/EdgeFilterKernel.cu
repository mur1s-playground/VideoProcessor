#include "EdgeFilterKernel.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

#include "math.h"

#include "Logger.h"

__constant__ float edge_filter_k[9] = {
		-1 / 8.0f, -1 / 8.0f, -1 / 8.0f,
		-1 / 8.0f, 1		, -1 / 8.0f,
		-1 / 8.0f, -1 / 8.0f, -1 / 8.0f
};

__global__ void edge_filter_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * channels) {
		int row = i / (channels * width);
		int col = (i % (channels * width)) / channels;
		int channel = (i % (channels * width)) % channels;

		float value = 0.0f;
		
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				if (row + y >= 0 && row + y < height && col + x >= 0 && col + x < width) {
					float base_val = ((float)src[(row + y) * width * channels + (col + x) * channels + channel]);
					value += ((edge_filter_k[(y + 1) * 3 + x + 1]) * base_val);
				}
			}
		}
		if (value > 255.0f) value = 255.0f;
		
		dst[i] = (unsigned char)value;	
	}
}

void edge_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * channels + threadsPerBlock - 1) / threadsPerBlock;
	edge_filter_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels);
	cudaStreamSynchronize(cuda_streams[3]);
}
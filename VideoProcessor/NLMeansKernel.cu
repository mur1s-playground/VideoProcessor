#include "NLMeansKernel.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

#include "math.h"

__forceinline__ __device__ double nl_means_kernel_region_weight(const int region_size, const unsigned char* src, const unsigned int i, const int width, const int height, const int channels) {
	int row = i / (channels * width);
	int col = (i % (channels * width)) / channels;
	int channel = (i % (channels * width)) % channels;
	double b_p = 0;
	double r_n = 0;
	for (int rs_r = -region_size/2; rs_r <= region_size/2; rs_r++) {
		for (int rs_c = -region_size/2; rs_c <= region_size/2; rs_c++) {
			if (row + rs_r >= 0 && row + rs_r < height &&
				col + rs_c >= 0 && col + rs_c < width) {
				int idx = (row + rs_r) * width * channels + (col + rs_c) * channels + channel;
				b_p += ((double)src[idx]) / 255.0;
				r_n += 1.0f;
			}
		}
	}
	b_p /= r_n;
	return b_p;
}

__global__ void nl_means_kernel(const int search_window_size, const int region_size, const float filtering_param, const unsigned char *src, const int width, const int height, const int channels, unsigned char *dest) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * channels) {
		int row = i / (channels * width);
		int col = (i % (channels * width)) / channels;
		int channel = (i % (channels * width)) % channels;

		double b_p = nl_means_kernel_region_weight(region_size, src, i, width, height, channels);
		
		double c_p = 0.0f;

		double u_p = 0.0f;
		for (int sw_r = -search_window_size/2; sw_r <= search_window_size/2; sw_r++) {
			for (int sw_c = -search_window_size / 2; sw_c <= search_window_size/2; sw_c++) {
				if (row + sw_r >= 0 && row + sw_r < height &&
					col + sw_c >= 0 && col + sw_c < width) {
					int idx = (row + sw_r) * width * channels + (col + sw_c) * channels + channel;
					double v_q = ((double)src[idx]) / 255.0;
					double b_q = nl_means_kernel_region_weight(region_size, src, idx, width, height, channels);
					double f_p_q = exp(-pow(b_q - b_p, 2) / pow(filtering_param, 2));
					c_p += f_p_q;
					u_p += v_q * f_p_q;
				}
			}
		}
		if (c_p > 0.0f) {
			u_p /= c_p;
		}
		dest[i] = (unsigned char)(u_p * 255.0);
	}
}

void nl_means_kernel_launch(const int search_window_size, const int region_size, const float filtering_param, const unsigned char* src, const int width, const int height, const int channels, unsigned char* dest) {
	//cudaError_t err = cudaSuccess;

	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * channels + threadsPerBlock - 1) / threadsPerBlock;
	nl_means_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[1] >> > (search_window_size, region_size, filtering_param, src, width, height, channels, dest);
	cudaStreamSynchronize(cuda_streams[1]);
}

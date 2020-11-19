#include "PaletteFilterKernel.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

#include "math.h"

#include "Logger.h"

__global__ void palette_filter_kernel(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int palette_bucket_quantization_size, const int palette_bucket_quantization_dim, const int palette_bucket_dimension_size, const float* kernel, const int* bucket_counts_ps) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height) {
		int row = i / width;
		int col = i % width;

		float closest = 255*3.0f;
		int closest_id = -1;

		int rgb[3] = { src[(row)*width * channels + (col)*channels] , src[(row)*width * channels + (col)*channels + 1] , src[(row)*width * channels + (col)*channels + 2] };
		int bucket_number = (rgb[0]/ palette_bucket_quantization_size * (palette_bucket_quantization_dim * palette_bucket_quantization_dim) + rgb[1]/palette_bucket_quantization_size * palette_bucket_quantization_dim + rgb[2]/palette_bucket_quantization_size) / palette_bucket_dimension_size;

		for (int k_s = bucket_counts_ps[bucket_number]; k_s < bucket_counts_ps[bucket_number+1]; k_s++) {
			float value = 0.0f;

			for (int ch = 0; ch < 3; ch++) {
				float val = (float)rgb[ch] - kernel[k_s * 3 + ch];
				value += (val * val);
			}
			value = sqrtf(value);
			if (value < closest) {
				closest = value;
				closest_id = k_s;
			}
		}

		for (int channel = 0; channel < channels; channel++) {
			dst[(row)*width * channels + (col)*channels + channel] = (unsigned char) kernel[closest_id * 3 + channel];
		}
	}
}

void palette_filter_kernel_launch(const unsigned char* src, unsigned char* dst, const int width, const int height, const int channels, const int palette_bucket_quantization_size, const int palette_bucket_quantization_dim, const int palette_bucket_dimension_size, const float* kernel, const int* bucket_counts_ps) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
	palette_filter_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[3] >> > (src, dst, width, height, channels, palette_bucket_quantization_size, palette_bucket_quantization_dim, palette_bucket_dimension_size, kernel, bucket_counts_ps);
	cudaStreamSynchronize(cuda_streams[3]);
}
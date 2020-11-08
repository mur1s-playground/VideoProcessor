#include "AudioVisualKernel.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

__global__ void gpu_audiovisual_kernel(const unsigned char* src, unsigned char* dst, const int src_width, const int src_height, const int src_channels, const float value1, const float value2, const float value3, const float value4, const float value5, const float value6, const float value7) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < src_width * src_height * src_channels) {
		int row = i / (src_channels * src_width);
		int col = (i % (src_channels * src_width)) / src_channels;
		int channel = (i % (src_channels * src_width)) % src_channels;

		float value = 0.0f;

		const unsigned char* zero = src;
		const unsigned char* full = &src[8 * src_width * src_height * src_channels];

		value = (float)full[i] - (float)src[i];
		float weight_norm = 0.0f;
		
		float weights[7];
		for (int dims = 1; dims < 8; dims++) {
			const unsigned char* dim = &src[dims * src_width * src_height * src_channels];
			if (value == 0) {
				weights[dims - 1] = 0;
			} else {
				weights[dims - 1] = ((float)dim[i] - (float)src[i]) / value;
			}
			weight_norm += weights[dims - 1];
		}
		
		if (weight_norm > 0) {
			float total_weight = 0.0f;
			for (int dims = 1; dims < 8; dims++) {
				weights[dims - 1] /= weight_norm;
				if (dims == 1) {
					weights[dims - 1] *= value1;
				}
				else if (dims == 2) {
					weights[dims - 1] *= value2;
				}
				else if (dims == 3) {
					weights[dims - 1] *= value3;
				}
				else if (dims == 4) {
					weights[dims - 1] *= value4;
				}
				else if (dims == 5) {
					weights[dims - 1] *= value5;
				}
				else if (dims == 6) {
					weights[dims - 1] *= value6;
				}
				else if (dims == 7) {
					weights[dims - 1] *= value7;
				}
				//weights[dims - 1] *= values[dims-1];
				total_weight += weights[dims - 1];
			}

			float result = (float)src[i] + total_weight * value;
			if (result > 255.0f) result = 255.0f;
			dst[i] = (unsigned char)result;
		} else {
			dst[i] = src[i];
		}
	}
}

void gpu_audiovisual_kernel_launch(const unsigned char* src, unsigned char* dst, const int src_width, const int src_height, const int src_channels, const float value1, const float value2, const float value3, const float value4, const float value5, const float value6, const float value7) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (src_width * src_height * src_channels + threadsPerBlock - 1) / threadsPerBlock;
	gpu_audiovisual_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[1] >> > (src, dst, src_width, src_height, src_channels, value1, value2, value3, value4, value5, value6, value7);
	cudaStreamSynchronize(cuda_streams[1]);
}
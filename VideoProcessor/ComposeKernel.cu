#include "ComposeKernel.h"

#include "cuda_runtime.h"

#include "CUDAStreamHandler.h"

#include "math.h"

__forceinline__
__device__ float interpixel(const unsigned char* frame, const unsigned int width, const unsigned int height, const unsigned int channels, float x, float y, const int c) {
    int x_i = (int)x;
    int y_i = (int)y;
    x -= x_i;
    y -= y_i;

    unsigned char value_components[4];
    value_components[0] = frame[y_i * (width * channels) + x_i * channels + c];
    if (x > 0) {
        if (x_i + 1 < width) {
            value_components[1] = frame[y_i * (width * channels) + (x_i + 1) * channels + c];
        }
        else {
            x = 0.0f;
        }
    }
    if (y > 0) {
        if (y_i + 1 < height) {
            value_components[2] = frame[(y_i + 1) * (width * channels) + x_i * channels + c];
            if (x > 0) {
                value_components[3] = frame[(y_i + 1) * (width * channels) + (x_i + 1) * channels + c];
            }
        }
        else {
            y = 0.0f;
        }
    }

    float m_0 = 4.0f / 16.0f;
    float m_1 = 4.0f / 16.0f;
    float m_2 = 4.0f / 16.0f;
    float m_3 = 4.0f / 16.0f;
    float tmp, tmp2;
    if (x <= 0.5f) {
        tmp = ((0.5f - x) / 0.5f) * m_1;
        m_0 += tmp;
        m_1 -= tmp;
        m_2 += tmp;
        m_3 -= tmp;
    }
    else {
        tmp = ((x - 0.5f) / 0.5f) * m_0;
        m_0 -= tmp;
        m_1 += tmp;
        m_2 -= tmp;
        m_3 += tmp;
    }
    if (y <= 0.5f) {
        tmp = ((0.5f - y) / 0.5f) * m_2;
        tmp2 = ((0.5f - y) / 0.5f) * m_3;
        m_0 += tmp;
        m_1 += tmp2;
        m_2 -= tmp;
        m_3 -= tmp2;
    }
    else {
        tmp = ((y - 0.5f) / 0.5f) * m_0;
        tmp2 = ((y - 0.5f) / 0.5f) * m_1;
        m_0 -= tmp;
        m_1 -= tmp2;
        m_2 += tmp;
        m_3 += tmp2;
    }
    float value = m_0 * value_components[0] + m_1 * value_components[1] + m_2 * value_components[2] + m_3 * value_components[3];
    return value;
}

__global__ void compose_kernel(const unsigned char* src, const int src_width, const int src_height, const int src_channels, const int dx, const int dy, const int crop_x1, const int crop_x2, const int crop_y1, const int crop_y2, const int width, const int height, unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * src_channels) {
		int base_row = (i / (src_channels * width));
		int base_col = ((i % (src_channels * width)) / src_channels);
		float step_size_width = (crop_x2 - crop_x1) /(float) width;
		float step_size_height = (crop_y2 - crop_y1) /(float) height;

        float step_size_w = max(1.0f, step_size_width);
        float step_size_h = max(1.0f, step_size_height);

		float src_row = crop_y1 + base_row*step_size_height;
		float src_col = crop_x1 + base_col*step_size_width;
		int src_channel = (i % (src_channels * width)) % src_channels;

        int dst_row = dy + base_row;
		int dst_col = dx + base_col;
		int dst_channel = src_channel;

		if (dst_channels == src_channels) {
			if (dst_row >= 0 && dst_row < dst_height && dst_col >= 0 && dst_col < dst_width && dst_channel < dst_channels) {
                float value = 0.0f;
                float v_c = 0.0f;
                for (int sf_h = 0; sf_h < step_size_h; sf_h++) {
                    for (int sf_w = 0; sf_w < step_size_w; sf_w++) {
                        value += interpixel(src, src_width, src_height, src_channels, src_col + sf_w, src_row + sf_h, src_channel);
                        v_c++;
                    }
                }
                dst[dst_row * (dst_width * dst_channels) + dst_col * dst_channels + dst_channel] = (unsigned char)(value / v_c);
			}
		} else if (dst_channels == 3 && src_channels == 4) {
			if (dst_row >= 0 && dst_row < dst_height && dst_col >= 0 && dst_col < dst_width && dst_channel < dst_channels) {
                float alpha_value = 0.0f;
                float value = 0.0f;
                float v_c = 0.0f;
                for (int sf_h = 0; sf_h < step_size_h; sf_h++) {
                    for (int sf_w = 0; sf_w < step_size_w; sf_w++) {
                        alpha_value += interpixel(src, src_width, src_height, src_channels, src_col + sf_w, src_row + sf_h, 3);
                        value += interpixel(src, src_width, src_height, src_channels, src_col + sf_w, src_row + sf_h, src_channel);
                        v_c++;
                    }
                }
                float res_val = ((255.0f - (alpha_value/v_c)) / 255.0f) * dst[dst_row * (dst_width * dst_channels) + dst_col * dst_channels + dst_channel] + ((alpha_value/v_c) / 255.0f) * (value / v_c);
                dst[dst_row * (dst_width * dst_channels) + dst_col * dst_channels + dst_channel] = (unsigned char) res_val;
			}
		}
	}
}

void compose_kernel_launch(const unsigned char* src, const int src_width, const int src_height, const int src_channels, const int dx, const int dy, const int crop_x1, const int crop_x2, const int crop_y1, const int crop_y2, const int width, const int height, unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * src_channels + threadsPerBlock - 1) / threadsPerBlock;
	compose_kernel << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[1] >> > (src, src_width, src_height, src_channels, dx, dy, crop_x1, crop_x2, crop_y1, crop_y2, width, height, dst, dst_width, dst_height, dst_channels);
	cudaStreamSynchronize(cuda_streams[1]);
}

__global__ void compose_kernel_set_zero(unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dst_width * dst_height * dst_channels) {
		dst[i] = 0;
	}
}

void compose_kernel_set_zero_launch(unsigned char* dst, const int dst_width, const int dst_height, const int dst_channels) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (dst_width * dst_height * dst_channels + threadsPerBlock - 1) / threadsPerBlock;
	compose_kernel_set_zero << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[1] >> > (dst, dst_width, dst_height, dst_channels);
	//cudaStreamSynchronize(cuda_streams[1]);
}

__global__ void compose_kernel_rgb_alpha_merge(const unsigned char* src_rgb, const unsigned char* src_alpha, unsigned char* dst, const int width, const int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < width * height * 4) {
		int dst_row = i / (4 * width);
		int dst_col = (i % (4 * width)) / 4;
		int dst_channel = (i % (4 * width)) % 4;

		int src_idx_base = dst_row * (3 * width) + dst_col * 3;
		if (dst_channel < 3) {
			dst[i] = src_rgb[src_idx_base + dst_channel];
		} else if (dst_channel == 3) {
			dst[i] = src_alpha[src_idx_base];
		}
	}
}

void compose_kernel_rgb_alpha_merge_launch(const unsigned char* src_rgb, const unsigned char* src_alpha, unsigned char* dst, const int width, const int height) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (width * height * 4 + threadsPerBlock - 1) / threadsPerBlock;
	compose_kernel_rgb_alpha_merge << <blocksPerGrid, threadsPerBlock, 0, cuda_streams[1] >> > (src_rgb, src_alpha, dst, width, height);
	cudaStreamSynchronize(cuda_streams[1]);
}
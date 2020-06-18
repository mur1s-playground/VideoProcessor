#pragma once

#include "VideoSource.h"

struct gpu_gaussian_blur {
	int kernel_size;
	float a, b, c;

	float* device_kernel;
	float norm_kernel;

	struct video_source* vs_in;
	struct gpu_memory_buffer* gmb_out;
};

void gpu_gaussian_blur_init(struct gpu_gaussian_blur* gb, const int kernel_size, const float a, const float b, const float c);
void gpu_gaussian_blur_edit(struct gpu_gaussian_blur* gb, const int kernel_size, const float a, const float b, const float c);

DWORD* gpu_gaussian_blur_loop(LPVOID args);

void gpu_gaussian_blur_externalise(struct application_graph_node* agn, string& out_str);
void gpu_gaussian_blur_load(struct gpu_gaussian_blur* gb, ifstream& in_f);
void gpu_gaussian_blur_destroy(struct application_graph_node* agn);
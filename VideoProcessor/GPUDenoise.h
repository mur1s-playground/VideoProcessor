#pragma once

#include <windows.h>

#include "GPUMemoryBuffer.h"
#include "VideoSource.h"

struct gpu_denoise {
	int search_window_size;
	int region_size;
	float filtering_param;

	struct video_source* vs_in;
	struct gpu_memory_buffer* gmb_out;
};

void gpu_denoise_init(struct gpu_denoise* gd, int search_window_size, int region_size, float filtering_param);
DWORD* gpu_denoise_loop(LPVOID args);

void gpu_denoise_externalise(struct application_graph_node* agn, string& out_str);
void gpu_denoise_load(struct gpu_denoise* gd, ifstream& in_f);
void gpu_denoise_destroy(struct application_graph_node* agn);
#pragma once

#include <windows.h>

#include "GPUMemoryBuffer.h"
#include "VideoSource.h"

struct vector3uc {
	unsigned char r, g, b;
};

struct gpu_palette_filter {
	int palette_size[2];
	float* device_palette[2];
	bool device_palette_switch;

	vector<float> palette;

	/* AUTOMATIC PALETTE GENERATION */
	float palette_auto_time;
	int palette_auto_size;
	int palette_auto_bucket_count;
	int palette_auto_quantization_size;

	float palette_auto_timer;
	/*------------------------------*/

	struct video_source* vs_in;
	struct gpu_memory_buffer* gmb_out;
};

void gpu_palette_filter_init(struct gpu_palette_filter* gpf, float palette_auto_time, int palette_auto_size, int palette_auto_bucket_count, int palette_auto_quantization_size);
DWORD* gpu_palette_filter_loop(LPVOID args);

void gpu_palette_filter_externalise(struct application_graph_node* agn, string& out_str);
void gpu_palette_filter_load(struct gpu_palette_filter* gpf, ifstream& in_f);
void gpu_palette_filter_destroy(struct application_graph_node* agn);

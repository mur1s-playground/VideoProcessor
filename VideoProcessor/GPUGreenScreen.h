#pragma once

#include "VideoSource.h"
#include "Vector3.h"

struct gpu_green_screen {
	struct vector3<unsigned char>	rgb;
	float							threshold;

	struct video_source* vs_in;
	struct gpu_memory_buffer* gmb_out;
};

void gpu_green_screen_init(struct gpu_green_screen* gb, const vector3<unsigned char> rgb, const float threshold);

DWORD* gpu_green_screen_loop(LPVOID args);

void gpu_green_screen_externalise(struct application_graph_node* agn, string& out_str);
void gpu_green_screen_load(struct gpu_green_screen* gb, ifstream& in_f);
void gpu_green_screen_destroy(struct application_graph_node* agn);
#pragma once

#include <windows.h>

#include "GPUMemoryBuffer.h"
#include "VideoSource.h"

struct gpu_edge_filter {
	struct video_source* vs_in;
	struct gpu_memory_buffer* gmb_out;
};

void gpu_edge_filter_init(struct gpu_edge_filter* gef);
DWORD* gpu_edge_filter_loop(LPVOID args);

void gpu_edge_filter_externalise(struct application_graph_node* agn, string& out_str);
void gpu_edge_filter_load(struct gpu_edge_filter* gef, ifstream& in_f);
void gpu_edge_filter_destroy(struct application_graph_node* agn);
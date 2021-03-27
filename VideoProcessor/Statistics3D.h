#pragma once

#include "SharedMemoryBuffer.h"

#include "Statistic.h"

struct statistics_3d {
	struct shared_memory_buffer*		smb_shared_state;

	struct statistic_heatmap			heatmap_3d;
	struct statistic_vectorfield_3d		movement_vectorfield_3d;

	struct video_source*				vs_out;
};

void statistics_3d_init(struct statistics_3d* s3d);
DWORD* statistics_3d_loop(LPVOID args);
void statistics_3d_externalise(struct application_graph_node* agn, string& out_str);
void statistics_3d_load(struct statistics_3d* s3d, ifstream& in_f);
void statistics_3d_destroy(struct application_graph_node* agn);
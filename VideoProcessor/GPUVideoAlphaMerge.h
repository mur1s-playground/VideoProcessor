#pragma once

#include "VideoSource.h"
#include "SharedMemoryBuffer.h"

struct gpu_video_alpha_merge {
	string name;

	bool sync_prio_rgb;

	struct video_source* vs_rgb;

	struct video_source* vs_alpha;
	int	channel_id;

	struct video_source* vs_out;
};

void gpu_video_alpha_merge_init(struct gpu_video_alpha_merge* vam, bool sync_prio_rgb, int alpha_id);
DWORD* gpu_video_alpha_merge_loop(LPVOID args);

void gpu_video_alpha_merge_externalise(struct application_graph_node* agn, string& out_str);
void gpu_video_alpha_merge_load(struct gpu_video_alpha_merge* vam, ifstream& in_f);
void gpu_video_alpha_merge_destroy(struct application_graph_node* agn);
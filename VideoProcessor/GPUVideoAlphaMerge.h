#pragma once

#include "VideoSource.h"
#include "SharedMemoryBuffer.h"

struct gpu_video_alpha_merge {
	string name;
	struct video_source* vs_rgb;

	struct video_source* vs_alpha;
	int	channel_id;

	struct video_source* vs_out;
};

void gpu_video_alpha_merge_init(struct gpu_video_alpha_merge* vam, int alpha_id);
DWORD* gpu_video_alpha_merge_loop(LPVOID args);

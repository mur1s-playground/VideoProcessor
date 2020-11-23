#pragma once

#include "VideoSource.h"

struct gpu_motion_blur {
	int frame_count;
	int weight_dist_type;
	float frame_id_weight_center;
	float a, b, c;
	float precision;
	
	bool calc_err;

	struct video_source* vs_in;
	struct gpu_memory_buffer* gmb_out;
};

void gpu_motion_blur_init(struct gpu_motion_blur* mb, int frame_count, int weight_dist_type, float frame_id_weight_center, float center_weight, float precision);
DWORD* gpu_motion_blur_loop(LPVOID args);

void gpu_motion_blur_calculate_weights(struct gpu_motion_blur* mb);

void gpu_motion_blur_externalise(struct application_graph_node* agn, string& out_str);
void gpu_motion_blur_load(struct gpu_motion_blur* mb, ifstream& in_f);
void gpu_motion_blur_destroy(struct application_graph_node* agn);
#pragma once

#include "VideoSource.h"
#include "ApplicationGraph.h"

struct gpu_composer_element {
	bool sync_prio;

	int delay;
	int dx, dy, crop_x1, crop_x2, crop_y1, crop_y2;

	float scale;
	int width, height;

	struct video_source* vs_in;
};

void gpu_composer_element_init(struct gpu_composer_element *gce);
void gpu_composer_element_externalise(struct application_graph_node* agn, string& out_str);
void gpu_composer_element_load(struct gpu_composer_element* gce, ifstream& in_f);
void gpu_composer_element_destroy(struct application_graph_node* agn);

#pragma once

#include "VideoSource.h"
#include "ApplicationGraph.h"

struct gpu_composer_element {
	int dx, dy, crop_x1, crop_x2, crop_y1, crop_y2;

	float scale;
	int width, height;

	struct video_source* vs_in;
};

void gpu_composer_element_init(struct gpu_composer_element *gce);

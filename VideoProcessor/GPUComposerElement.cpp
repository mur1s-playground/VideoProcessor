#include "GPUComposerElement.h"

void gpu_composer_element_init(struct gpu_composer_element* gce) {
	gce->dx = 0;
	gce->dy = 0;

	gce->height = 0;
	gce->width = 0;
	gce->crop_x1 = 0;
	gce->crop_x2 = 0;
	gce->crop_y1 = 0;
	gce->crop_y2 = 0;

	gce->vs_in = nullptr;
}
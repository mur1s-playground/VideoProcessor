#include "GPUComposerElement.h"

#include <sstream>
#include <fstream>

void gpu_composer_element_init(struct gpu_composer_element* gce) {
	gce->sync_prio = false;
	gce->delay = 0;
	gce->dx = 0;
	gce->dy = 0;

	gce->height = 0;
	gce->width = 0;
	gce->scale = 1.0f;
	gce->crop_x1 = 0;
	gce->crop_x2 = 0;
	gce->crop_y1 = 0;
	gce->crop_y2 = 0;

	gce->vs_in = nullptr;
}


void gpu_composer_element_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_composer_element* gce = (struct gpu_composer_element*)agn->component;

	stringstream s_out;
	if (gce->sync_prio) {
		s_out << 1 << std::endl;
	} else {
		s_out << 0 << std::endl;
	}
	s_out << gce->delay << std::endl;

	s_out << gce->dx << std::endl;
	s_out << gce->dy << std::endl;
	
	s_out << gce->crop_x1 << std::endl;
	s_out << gce->crop_x2 << std::endl;
	s_out << gce->crop_y1 << std::endl;
	s_out << gce->crop_y2 << std::endl;

	s_out << gce->scale << std::endl;

	out_str = s_out.str();
}

void gpu_composer_element_load(struct gpu_composer_element *gce, ifstream &in_f) {
	std::string line;
	std::getline(in_f, line);
	gce->sync_prio = (stoi(line) == 0);
	std::getline(in_f, line);
	gce->delay = stoi(line);
	std::getline(in_f, line);
	gce->dx = stoi(line);
	std::getline(in_f, line);
	gce->dy = stoi(line);
	std::getline(in_f, line);
	gce->crop_x1 = stoi(line);
	std::getline(in_f, line);
	gce->crop_x2 = stoi(line);
	std::getline(in_f, line);
	gce->crop_y1 = stoi(line);
	std::getline(in_f, line);
	gce->crop_y2 = stoi(line);
	std::getline(in_f, line);
	gce->scale = stof(line);

	gce->vs_in = nullptr;
}

void gpu_composer_element_destroy(struct application_graph_node* agn) {
	struct gpu_composer_element* gce = (struct gpu_composer_element*)agn->component;
	delete gce;
}
#pragma once

#include <windows.h>

#include <vector>
#include <string>

#include "GPUComposerElement.h"
#include "VideoSource.h"

using namespace std;

struct gpu_composer {
	string name;

	vector<struct gpu_composer_element*> gce_ins;

	struct gpu_composer_element* gce_in_connector;

	struct video_source* vs_out;
};

void gpu_composer_init(struct gpu_composer* gc, const char* name);

void gpu_composer_on_input_connect(struct application_graph_node* agn, int input_id);
DWORD* gpu_composer_loop(LPVOID args);

void gpu_composer_externalise(struct application_graph_node* agn, string& out_str);
void gpu_composer_load(struct gpu_composer* gc, ifstream& in_f);
void gpu_composer_destroy(struct application_graph_node* agn);

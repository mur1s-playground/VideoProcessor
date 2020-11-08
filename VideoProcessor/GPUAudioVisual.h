#pragma once

#include "VideoSource.h"

using namespace std;

struct gpu_audiovisual {
	string name;

	vector<string> frame_names;

	Mat *mats_in;

	struct shared_memory_buffer* smb_in;
	struct gpu_memory_buffer* gmb_in;
	struct video_source* vs_out;
};

void gpu_audiovisual_init(struct gpu_audiovisual* gav, const char* name, vector<string> files_names);

void gpu_audiovisual_on_input_connect(struct application_graph_node* agn, int input_id);
void gpu_audiovisual_on_input_disconnect(struct application_graph_edge* edge);

DWORD* gpu_audiovisual_loop(LPVOID args);

void gpu_audiovisual_externalise(struct application_graph_node* agn, string& out_str);
void gpu_audiovisual_load(struct gpu_composer* gc, ifstream& in_f);
void gpu_audiovisual_destroy(struct application_graph_node* agn);
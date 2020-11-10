#pragma once

#include "VideoSource.h"
#include "AudioSource.h"

using namespace std;

struct gpu_audiovisual_dft {
	float norms[1024];
};

struct gpu_audiovisual {
	string name;
	int fps_target;
	int dft_size;
	float amplify;

	vector<string> frame_names;

	Mat *mats_in;

	struct audio_source* audio_source_in;
	struct gpu_memory_buffer* dft_out;
	struct gpu_memory_buffer* gmb_in;
	struct video_source* vs_out;

	struct gpu_audiovisual_dft dft;
};

void gpu_audiovisual_init(struct gpu_audiovisual* gav, const char* name, int dft_size);

void gpu_audiovisual_on_input_connect(struct application_graph_node* agn, int input_id);
void gpu_audiovisual_on_input_disconnect(struct application_graph_edge* edge);

DWORD* gpu_audiovisual_loop(LPVOID args);

void gpu_audiovisual_externalise(struct application_graph_node* agn, string& out_str);
void gpu_audiovisual_load(struct gpu_audiovisual* gav, ifstream& in_f);
void gpu_audiovisual_destroy(struct application_graph_node* agn);
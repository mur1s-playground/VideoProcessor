#pragma once

#include <windows.h>

#include <vector>
#include <string>

#include "CameraControl.h"
#include "GPUMemoryBuffer.h"
#include "VideoSource.h"

using namespace std;

struct camera_control_diagnostic {
	struct camera_control* cc;

	struct gpu_memory_buffer* cc_shared_state_gpu;
	struct video_source* vs_out;
};

void camera_control_diagnostic_init(struct camera_control_diagnostic* ccd);

void camera_control_diagnostic_on_input_connect(struct application_graph_node* agn, int input_id);

DWORD* camera_control_diagnostic_loop(LPVOID args);

void camera_control_diagnostic_externalise(struct application_graph_node* agn, string& out_str);
void camera_control_diagnostic_load(struct  camera_control_diagnostic* ccd, ifstream& in_f);
void camera_control_diagnostic_destroy(struct application_graph_node* agn);
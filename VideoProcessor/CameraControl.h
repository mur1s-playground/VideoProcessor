#pragma once

#include <windows.h>

#include <string>

#include "SharedMemoryBuffer.h"

using namespace std;

struct camera_control {
	struct shared_memory_buffer* smb_det;
};

void camera_control_init(struct camera_control* cc);

DWORD* camera_control_loop(LPVOID args);

void camera_control_externalise(struct application_graph_node* agn, string& out_str);
void camera_control_load(struct camera_control* cc, ifstream& in_f);
void camera_control_destroy(struct application_graph_node* agn);

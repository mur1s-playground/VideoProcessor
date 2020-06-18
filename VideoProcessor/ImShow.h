#pragma once

#include "VideoSource.h"

struct im_show {
	string name;
	struct video_source* vs;
};

void im_show_init(struct im_show* is, const char* name);
DWORD* im_show_loop(LPVOID args);

void im_show_externalise(struct application_graph_node* agn, string& out_str);
void im_show_load(struct im_show* is, ifstream& in_f);
void im_show_destroy(struct application_graph_node* agn);
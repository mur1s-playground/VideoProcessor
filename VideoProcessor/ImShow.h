#pragma once

#include "VideoSource.h"

struct im_show {
	string name;
	struct video_source* vs;
};

void im_show_init(struct im_show* is, const char* name);
DWORD* im_show_loop(LPVOID args);
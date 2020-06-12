#pragma once

#include <opencv2/videoio.hpp>

#include "SharedMemoryBuffer.h"
#include "GPUMemoryBuffer.h"

using namespace cv;
using namespace std;

struct video_source {
	VideoCapture video_capture;
	string name;

	int video_width;
	int video_height;
	int video_channels;

	bool is_open;
	bool read_video_capture;
	bool do_copy;
	bool direction_smb_to_gmb;

	Mat* mats;

	struct shared_memory_buffer *smb;
	int smb_last_used_id;
	int smb_framecount;

	int smb_size_req;

	struct gpu_memory_buffer* gmb;
};

//void video_source_set_meta(struct video_source* vs);
void video_source_init(struct video_source* vs, int device_id);
void video_source_init(struct video_source* vs, const char* path);

void video_source_on_input_connect(struct application_graph_node* agn, int input_id);
DWORD* video_source_loop(LPVOID args);

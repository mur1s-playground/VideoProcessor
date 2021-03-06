#pragma once

#include <opencv2/dnn.hpp>
#include <vector>

#include "VideoSource.h"
#include "SharedMemoryBuffer.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

struct mask_rcnn_detection {
	int class_id;
	float score;
	int x1, x2, y1, y2;
};

struct mask_rcnn {
	Net net;

	vector<string> net_classes_available;
	vector<string> net_classes_active;

	float net_conf_threshold;
	float net_mask_threshold;
	float scale;
	bool draw_box;
	bool draw_mask;
	
	struct video_source* v_src_in;
	struct video_source* v_src_out;

	struct shared_memory_buffer* smb_det;
	
	Mat blob;
};

void mask_rcnn_init(struct mask_rcnn *mrcnn);
DWORD* mask_rcnn_loop(LPVOID args);

void mask_rcnn_externalise(struct application_graph_node* agn, string& out_str);
void mask_rcnn_load(struct mask_rcnn* mrcnn, ifstream& in_f);
void mask_rcnn_destroy(struct application_graph_node* agn);
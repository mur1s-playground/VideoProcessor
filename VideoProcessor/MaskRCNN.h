#pragma once

#include <opencv2/dnn.hpp>
#include <vector>

#include "VideoSource.h"
#include "SharedMemoryBuffer.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

struct mask_rcnn {
	Net net;

	vector<string> net_classes_available;
	vector<string> net_classes_active;

	float net_conf_threshold;
	float net_mask_threshold;
	
	struct video_source* v_src_in;
	struct video_source* v_src_out;
	
	Mat blob;
};

void mask_rcnn_init(struct mask_rcnn *mrcnn);
DWORD* mask_rcnn_loop(LPVOID args);


#include "ImShow.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "Logger.h"

void im_show_init(struct im_show *is, const char *name) {
	stringstream ss_name;
	ss_name << name;
	is->name = ss_name.str();
}

DWORD* im_show_loop(LPVOID args) {
	struct application_graph_node *agn = (struct application_graph_node*)args;
	struct im_show* is = (struct im_show*)agn->component;

	if (is->vs == nullptr) return NULL;
	if (is->vs->smb == nullptr) return NULL;
	if (is->vs->mats == nullptr) return NULL;

	int last_frame = -1;
	namedWindow(is->name.c_str(), WINDOW_AUTOSIZE);
	while (agn->process_run) {
		shared_memory_buffer_try_r(is->vs->smb, is->vs->smb_framecount, true, 8);
										      //slots																	//rw-locks									      //meta
		int next_frame = is->vs->smb->p_buf_c[is->vs->smb_framecount * is->vs->video_channels * is->vs->video_height * is->vs->video_width + ((is->vs->smb_framecount + 1) * 2)];
		shared_memory_buffer_release_r(is->vs->smb, is->vs->smb_framecount);

		if (next_frame != last_frame) {
			shared_memory_buffer_try_r(is->vs->smb, next_frame, true, 8);
			cv::imshow(is->name.c_str(), is->vs->mats[next_frame]);
			shared_memory_buffer_release_r(is->vs->smb, next_frame);
		}
		cv::waitKey(16);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}
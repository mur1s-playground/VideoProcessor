#include "ImShow.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "Logger.h"

#include <sstream>
#include <fstream>

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
		application_graph_tps_balancer_timer_start(agn);
		shared_memory_buffer_try_r(is->vs->smb, is->vs->smb_framecount, true, 8);
										      //slots																	//rw-locks									      //meta
		int next_frame = is->vs->smb->p_buf_c[is->vs->smb_framecount * is->vs->video_channels * is->vs->video_height * is->vs->video_width + ((is->vs->smb_framecount + 1) * 2)];
		shared_memory_buffer_release_r(is->vs->smb, is->vs->smb_framecount);

		if (next_frame != last_frame) {
			shared_memory_buffer_try_r(is->vs->smb, next_frame, true, 8);
			cv::imshow(is->name.c_str(), is->vs->mats[next_frame]);
			shared_memory_buffer_release_r(is->vs->smb, next_frame);
		}
		application_graph_tps_balancer_timer_stop(agn);
		int sleep_time = application_graph_tps_balancer_get_sleep_ms(agn);
		if (sleep_time > 0) cv::waitKey(sleep_time);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void im_show_externalise(struct application_graph_node* agn, string& out_str) {
	struct im_show* is = (struct im_show*)agn->component;

	stringstream s_out;
	s_out << is->name << std::endl;
	out_str = s_out.str();
}

void im_show_load(struct im_show* is, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	string name = line;
	
	im_show_init(is, name.c_str());
}

void im_show_destroy(struct application_graph_node* agn) {
	struct im_show* is = (struct im_show*)agn->component;
	delete is;
}
#include "GPUMotionBlur.h"

#include "MotionBlur.h"

#include "ApplicationGraph.h"

#include <sstream>
#include <fstream>

#include "MainUI.h"

void gpu_motion_blur_init(struct gpu_motion_blur* mb, int frame_count) {
	mb->frame_count = frame_count;
	mb->vs_in = nullptr;
	mb->gmb_out = nullptr;
}

DWORD* gpu_motion_blur_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_motion_blur* mb = (struct gpu_motion_blur*)agn->component;

	if (mb->vs_in == nullptr || mb->vs_in->gmb == nullptr) return NULL;
	int last_gpu_id = -1;
	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);
		
		gpu_memory_buffer_try_r(mb->vs_in->gmb, mb->vs_in->gmb->slots, true, 8);
		int next_gpu_id = mb->vs_in->gmb->p_rw[2 * (mb->vs_in->gmb->slots + 1)];
		gpu_memory_buffer_release_r(mb->vs_in->gmb, mb->vs_in->gmb->slots);
		if (next_gpu_id != last_gpu_id) {

			gpu_memory_buffer_try_r(mb->gmb_out, mb->gmb_out->slots, true, 8);
			int next_gpu_out_id = (mb->gmb_out->p_rw[2 * (mb->gmb_out->slots + 1)] + 1) % mb->gmb_out->slots;
			gpu_memory_buffer_release_r(mb->gmb_out, mb->gmb_out->slots);

			gpu_memory_buffer_try_rw(mb->gmb_out, next_gpu_out_id, true, 8);

			int frame_counter = 0;
			for (int i = next_gpu_id - mb->frame_count + 1; i <= next_gpu_id; i++) {
				int id = i;
				if (id < 0) {
					id += mb->vs_in->gmb->slots;
				}
				gpu_memory_buffer_try_r(mb->vs_in->gmb, id, true, 8);

				motion_blur_kernel_launch(&mb->vs_in->gmb->p_device[id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels], &mb->gmb_out->p_device[next_gpu_out_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels], mb->vs_in->video_width, mb->vs_in->video_height, mb->vs_in->video_channels, mb->frame_count, frame_counter, true);

				gpu_memory_buffer_release_r(mb->vs_in->gmb, id);

				frame_counter++;
			}

			gpu_memory_buffer_release_rw(mb->gmb_out, next_gpu_out_id);

			gpu_memory_buffer_try_rw(mb->gmb_out, mb->gmb_out->slots, true, 8);
			mb->gmb_out->p_rw[2 * (mb->gmb_out->slots + 1)] = next_gpu_out_id;
			gpu_memory_buffer_release_rw(mb->gmb_out, mb->gmb_out->slots);
			last_gpu_id = next_gpu_id;
		}
		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_motion_blur_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_motion_blur* mb = (struct gpu_motion_blur*)agn->component; 

	stringstream s_out;
	s_out << mb->frame_count << std::endl;
	
	out_str = s_out.str();
}

void gpu_motion_blur_load(struct gpu_motion_blur* mb, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	mb->frame_count = stoi(line);
	
	mb->vs_in = nullptr;
	mb->gmb_out = nullptr;
}

void gpu_motion_blur_destroy(struct application_graph_node* agn) {
	struct gpu_motion_blur* mb = (struct gpu_motion_blur*)agn->component;
	delete mb;
}

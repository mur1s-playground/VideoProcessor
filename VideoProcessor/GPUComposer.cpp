#include "GPUComposer.h"

#include <sstream>
#include <fstream>
#include <algorithm>

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "ComposeKernel.h"

#include "Logger.h"

void gpu_composer_init(struct gpu_composer* gc, const char* name) {
	stringstream ss_name;
	ss_name << name;
	gc->name = ss_name.str();

	gc->gce_in_connector = nullptr;
	gc->vs_out = nullptr;
}

void gpu_composer_on_input_connect(struct application_graph_node *agn, int input_id) {
	struct gpu_composer* gc = (struct gpu_composer*)agn->component;

	if (input_id == 0) {
		gc->gce_ins.push_back(gc->gce_in_connector);
	}
}

void gpu_composer_on_input_disconnect(struct application_graph_edge *edge) {
	struct gpu_composer* gc = (struct gpu_composer*)edge->to.first->component;
	int input_id = edge->to.second;
	if (input_id == 0) {
		struct gpu_composer_element* gce = (struct gpu_composer_element*)edge->from.first->component;
		vector<gpu_composer_element*>::iterator gce_ins_it = find(gc->gce_ins.begin(), gc->gce_ins.end(), gce);
		if (gce_ins_it != gc->gce_ins.end()) {
			gc->gce_ins.erase(gce_ins_it);
		}
	}
}

DWORD* gpu_composer_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_composer* gc = (struct gpu_composer*)agn->component;

	int prio_id = 0;
	for (int ic = 0; ic < gc->gce_ins.size(); ic++) {
		struct gpu_composer_element* current_gce = gc->gce_ins[ic];
		if (current_gce->delay == 1) {
			prio_id = ic;
			break;
		}
	}

	int next_frame_out = -1;
	unsigned long long sync_time = 0;
	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		next_frame_out = (next_frame_out + 1) % gc->vs_out->smb_framecount;
		
		gpu_memory_buffer_try_rw(gc->vs_out->gmb, next_frame_out, true, 8);
		
		compose_kernel_set_zero_launch(gc->vs_out->gmb->p_device + (next_frame_out * gc->vs_out->video_width * gc->vs_out->video_height * gc->vs_out->video_channels), gc->vs_out->video_width, gc->vs_out->video_height, gc->vs_out->video_channels);

		//prio
		struct gpu_composer_element* current_gce = gc->gce_ins[prio_id];
		struct video_source* current_vs = current_gce->vs_in;

		gpu_memory_buffer_try_r(current_vs->gmb, current_vs->gmb->slots, true, 8);
		int prio_input_id = current_vs->gmb->p_rw[2 * (current_vs->gmb->slots + 1)];
		gpu_memory_buffer_release_r(current_vs->gmb, current_vs->gmb->slots);

		gpu_memory_buffer_try_r(current_vs->gmb, prio_input_id, true, 8);
		sync_time = gpu_memory_buffer_get_time(current_vs->gmb, prio_input_id);

		for (int ic = 0; ic < gc->gce_ins.size(); ic++) {
			current_gce = gc->gce_ins[ic];
			current_vs = current_gce->vs_in;

			current_gce->width = (int)round((current_gce->crop_x2 - current_gce->crop_x1) * current_gce->scale);
			current_gce->height = (int)round((current_gce->crop_y2 - current_gce->crop_y1) * current_gce->scale);

			int current_input_id = 0;
			if (ic == prio_id) {
				current_input_id = prio_input_id;
			} else {
				gpu_memory_buffer_try_r(current_vs->gmb, current_vs->gmb->slots, true, 8);
				current_input_id = current_vs->gmb->p_rw[2 * (current_vs->gmb->slots + 1)];
				gpu_memory_buffer_release_r(current_vs->gmb, current_vs->gmb->slots);

				for (int td = 0; td < current_vs->gmb->slots; td++) {
					int tmp_id = current_input_id - td;
					if (tmp_id < 0) tmp_id += current_vs->gmb->slots;
					gpu_memory_buffer_try_r(current_vs->gmb, tmp_id, true, 8);
					if (gpu_memory_buffer_get_time(current_vs->gmb, tmp_id) <= sync_time) {
						current_input_id = tmp_id;
						break;
					}
					gpu_memory_buffer_release_r(current_vs->gmb, tmp_id);
				}
			}
			compose_kernel_launch(current_vs->gmb->p_device + (current_input_id * current_vs->video_width * current_vs->video_height * current_vs->video_channels), current_vs->video_width, current_vs->video_height, current_vs->video_channels, current_gce->dx, current_gce->dy, current_gce->crop_x1, current_gce->crop_x2, current_gce->crop_y1, current_gce->crop_y2, current_gce->width, current_gce->height, gc->vs_out->gmb->p_device + (next_frame_out * gc->vs_out->video_width * gc->vs_out->video_height * gc->vs_out->video_channels), gc->vs_out->video_width, gc->vs_out->video_height, gc->vs_out->video_channels);
			gpu_memory_buffer_release_r(current_vs->gmb, current_input_id);
		}
		gpu_memory_buffer_set_time(gc->vs_out->gmb, next_frame_out, sync_time);
		gpu_memory_buffer_release_rw(gc->vs_out->gmb, next_frame_out);

		gpu_memory_buffer_try_rw(gc->vs_out->gmb, gc->vs_out->gmb->slots, true, 8);
		gc->vs_out->gmb->p_rw[2 * (gc->vs_out->gmb->slots + 1)] = next_frame_out;
		gpu_memory_buffer_release_rw(gc->vs_out->gmb, gc->vs_out->gmb->slots);

		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_composer_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_composer* gc = (struct gpu_composer*)agn->component;

	stringstream s_out;
	s_out << gc->name << std::endl;
	
	out_str = s_out.str();
}

void gpu_composer_load(struct gpu_composer* gc, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	gc->name = line;

	gc->gce_in_connector = nullptr;
	gc->vs_out = nullptr;
}

void gpu_composer_destroy(struct application_graph_node* agn) {
	struct gpu_composer* gc = (struct gpu_composer*)agn->component;
	delete gc;
}
#include "EdgeFilterKernel.h"

#include "GPUEdgeFilter.h"

#include "MainUI.h"
#include "ApplicationGraph.h"

#include <sstream>
#include <fstream>

void gpu_edge_filter_init(struct gpu_edge_filter* gef, float amplify) {
	gef->amplify = amplify;

	gef->vs_in = nullptr;
	gef->gmb_out = nullptr;
}

DWORD* gpu_edge_filter_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_edge_filter* gef = (struct gpu_edge_filter*)agn->component;

	if (gef->vs_in == nullptr || gef->vs_in->gmb == nullptr) return NULL;
	int last_gpu_id = -1;
	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		gpu_memory_buffer_try_r(gef->vs_in->gmb, gef->vs_in->gmb->slots, true, 8);
		int next_gpu_id = gef->vs_in->gmb->p_rw[2 * (gef->vs_in->gmb->slots + 1)];
		gpu_memory_buffer_release_r(gef->vs_in->gmb, gef->vs_in->gmb->slots);
		if (next_gpu_id != last_gpu_id) {
			gpu_memory_buffer_try_r(gef->gmb_out, gef->gmb_out->slots, true, 8);
			int next_gpu_out_id = (gef->gmb_out->p_rw[2 * (gef->gmb_out->slots + 1)] + 1) % gef->gmb_out->slots;
			gpu_memory_buffer_release_r(gef->gmb_out, gef->gmb_out->slots);

			gpu_memory_buffer_try_r(gef->vs_in->gmb, next_gpu_id, true, 8);
			gpu_memory_buffer_try_rw(gef->gmb_out, next_gpu_out_id, true, 8);

			edge_filter_kernel_launch(&gef->vs_in->gmb->p_device[next_gpu_id * gef->vs_in->video_channels * gef->vs_in->video_width * gef->vs_in->video_height], &gef->gmb_out->p_device[next_gpu_out_id * gef->vs_in->video_width * gef->vs_in->video_height], gef->vs_in->video_width, gef->vs_in->video_height, gef->vs_in->video_channels, gef->amplify);
			gpu_memory_buffer_set_time(gef->gmb_out, next_gpu_out_id, gpu_memory_buffer_get_time(gef->vs_in->gmb, next_gpu_id));

			gpu_memory_buffer_release_rw(gef->gmb_out, next_gpu_out_id);
			gpu_memory_buffer_release_r(gef->vs_in->gmb, next_gpu_id);

			gpu_memory_buffer_try_rw(gef->gmb_out, gef->gmb_out->slots, true, 8);
			gef->gmb_out->p_rw[2 * (gef->gmb_out->slots + 1)] = next_gpu_out_id;
			gpu_memory_buffer_release_rw(gef->gmb_out, gef->gmb_out->slots);
			last_gpu_id = next_gpu_id;
		}
		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_edge_filter_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_edge_filter* gef = (struct gpu_edge_filter*)agn->component;

	stringstream s_out;
	s_out << gef->amplify << std::endl;

	out_str = s_out.str();
}

void gpu_edge_filter_load(struct gpu_edge_filter* gef, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	gef->amplify = stof(line);

	gef->vs_in = nullptr;
	gef->gmb_out = nullptr;
}

void gpu_edge_filter_destroy(struct application_graph_node* agn) {
	struct gpu_edge_filter* gef = (struct gpu_edge_filter*)agn->component;
	delete gef;
}
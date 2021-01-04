#include "GPUGreenScreen.h"

#include "GreenScreenKernel.h"
#include "ApplicationGraph.h"
#include "MainUI.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

#include <sstream>
#include <fstream>

#include "Logger.h"

void gpu_green_screen_init(struct gpu_green_screen* gb, const vector3<unsigned char> rgb, const float threshold) {
	gb->rgb = rgb;
	gb->threshold = threshold;
	
	gb->vs_in = nullptr;
	gb->gmb_out = nullptr;
}

DWORD* gpu_green_screen_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_green_screen* mb = (struct gpu_green_screen*)agn->component;

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

			gpu_memory_buffer_try_r(mb->vs_in->gmb, next_gpu_id, true, 8);

			green_screen_kernel_launch(mb->vs_in->gmb->p_device + (next_gpu_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels), mb->gmb_out->p_device + (next_gpu_out_id * mb->vs_in->video_width * mb->vs_in->video_height * 4), mb->vs_in->video_width, mb->vs_in->video_height, 4, mb->rgb, mb->threshold);
			gpu_memory_buffer_set_time(mb->gmb_out, next_gpu_out_id, gpu_memory_buffer_get_time(mb->vs_in->gmb, next_gpu_id));

			gpu_memory_buffer_release_r(mb->vs_in->gmb, next_gpu_id);

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

void gpu_green_screen_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_green_screen* mb = (struct gpu_green_screen*)agn->component;

	stringstream s_out;
	s_out << (int)mb->rgb[0] << std::endl;
	s_out << (int)mb->rgb[1] << std::endl;
	s_out << (int)mb->rgb[2] << std::endl;
	s_out << mb->threshold << std::endl;

	out_str = s_out.str();
}

void gpu_green_screen_load(struct gpu_green_screen* gb, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	gb->rgb[0] = stoi(line);
	std::getline(in_f, line);
	gb->rgb[1] = stoi(line);
	std::getline(in_f, line);
	gb->rgb[2] = stoi(line);
	std::getline(in_f, line);
	gb->threshold = stof(line);

	gpu_green_screen_init(gb, gb->rgb, gb->threshold);
}

void gpu_green_screen_destroy(struct application_graph_node* agn) {
	struct gpu_green_screen* mb = (struct gpu_green_screen*)agn->component;	
	delete mb;
}
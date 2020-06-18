#include "GPUDenoise.h"

#include "MainUI.h"
#include "ApplicationGraph.h"
#include "NLMeansKernel.h"

#include <sstream>
#include <fstream>

void gpu_denoise_init(struct gpu_denoise* gd, int search_window_size, int region_size, float filtering_param) {
	gd->search_window_size = search_window_size;
	gd->region_size = region_size;
	gd->filtering_param = filtering_param;

	gd->vs_in = nullptr;
	gd->gmb_out = nullptr;
}

DWORD* gpu_denoise_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_denoise* gd = (struct gpu_denoise*)agn->component;

	if (gd->vs_in == nullptr || gd->vs_in->gmb == nullptr) return NULL;
	int last_gpu_id = -1;
	while (agn->process_run) {
		gpu_memory_buffer_try_r(gd->vs_in->gmb, gd->vs_in->gmb->slots, true, 8);
		int next_gpu_id = gd->vs_in->gmb->p_rw[2 * (gd->vs_in->gmb->slots + 1)];
		gpu_memory_buffer_release_r(gd->vs_in->gmb, gd->vs_in->gmb->slots);
		if (next_gpu_id != last_gpu_id) {
			gpu_memory_buffer_try_r(gd->gmb_out, gd->gmb_out->slots, true, 8);
			int next_gpu_out_id = (gd->gmb_out->p_rw[2 * (gd->gmb_out->slots + 1)] + 1) % gd->gmb_out->slots;
			gpu_memory_buffer_release_r(gd->gmb_out, gd->gmb_out->slots);

			gpu_memory_buffer_try_r(gd->vs_in->gmb, next_gpu_id, true, 8);
			gpu_memory_buffer_try_rw(gd->gmb_out, next_gpu_out_id, true, 8);

			nl_means_kernel_launch(gd->search_window_size, gd->region_size, gd->filtering_param, &gd->vs_in->gmb->p_device[next_gpu_id * gd->vs_in->video_channels * gd->vs_in->video_width * gd->vs_in->video_height], gd->vs_in->video_width, gd->vs_in->video_height, gd->vs_in->video_channels, &gd->gmb_out->p_device[next_gpu_out_id * gd->vs_in->video_channels * gd->vs_in->video_width * gd->vs_in->video_height]);

			gpu_memory_buffer_release_rw(gd->gmb_out, next_gpu_out_id);
			gpu_memory_buffer_release_r(gd->vs_in->gmb, next_gpu_id);

			gpu_memory_buffer_try_rw(gd->gmb_out, gd->gmb_out->slots, true, 8);
			gd->gmb_out->p_rw[2 * (gd->gmb_out->slots + 1)] = next_gpu_out_id;
			gpu_memory_buffer_release_rw(gd->gmb_out, gd->gmb_out->slots);
			last_gpu_id = next_gpu_id;
		}
		Sleep(16);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_denoise_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_denoise* gd = (struct gpu_denoise*)agn->component;

	stringstream s_out;
	s_out << gd->search_window_size << std::endl;
	s_out << gd->region_size << std::endl;

	s_out << gd->filtering_param << std::endl;
	
	out_str = s_out.str();
}

void gpu_denoise_load(struct gpu_denoise* gd, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	gd->search_window_size = stoi(line);
	std::getline(in_f, line);
	gd->region_size = stoi(line);
	std::getline(in_f, line);
	gd->filtering_param = stoi(line);
	
	gd->vs_in = nullptr;
	gd->gmb_out = nullptr;
}

void gpu_denoise_destroy(struct application_graph_node* agn) {
	struct gpu_denoise* gd = (struct gpu_denoise*)agn->component;
	delete gd;
}
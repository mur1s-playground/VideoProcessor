#include "GPUPaletteFilter.h"

#include "PaletteFilterKernel.h"
#include "ApplicationGraph.h"
#include "MainUI.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

#include <sstream>
#include <fstream>

#include "Logger.h"

void gpu_palette_filter_init(struct gpu_palette_filter* gpf) {
	gpf->palette_size = gpf->palette.size()/3;
	
	gpf->vs_in = nullptr;
	gpf->gmb_out = nullptr;

	cudaMalloc((void**)&gpf->device_palette, 3 * gpf->palette_size * sizeof(float));
	cudaMemcpyAsync(gpf->device_palette, gpf->palette.data(), 3 * gpf->palette_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaStreamSynchronize(cuda_streams[0]);
}

void gpu_palette_filter_edit(struct gpu_palette_filter* gpf) {
	gpf->palette_size = gpf->palette.size() / 3;
#
	cudaFree(gpf->device_palette);
	cudaMalloc((void**)&gpf->device_palette, 3 * gpf->palette_size * sizeof(float));
	cudaMemcpyAsync(gpf->device_palette, gpf->palette.data(), 3 * gpf->palette_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaStreamSynchronize(cuda_streams[0]);
}

DWORD* gpu_palette_filter_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_palette_filter* mb = (struct gpu_palette_filter*)agn->component;

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

			palette_filter_kernel_launch(mb->vs_in->gmb->p_device + (next_gpu_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels), mb->gmb_out->p_device + (next_gpu_out_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels), mb->vs_in->video_width, mb->vs_in->video_height, mb->vs_in->video_channels, mb->device_palette, mb->palette_size);

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

void gpu_palette_filter_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_palette_filter* mb = (struct gpu_palette_filter*)agn->component;

	stringstream s_out;

	for (int i = 0; i < mb->palette.size(); i++) {
		if (i > 0) {
			s_out << ",";
		}
		s_out << mb->palette[i];
	}
	out_str = s_out.str();
}

void gpu_palette_filter_load(struct gpu_palette_filter* gb, ifstream& in_f) {
	std::string line;
	
	std::getline(in_f, line);
	int start = 0;
	int end = line.find_first_of(",", start);
	while (end != std::string::npos) {
		gb->palette.push_back(stof(line.substr(start, end - start).c_str()));
		start = end + 1;
		end = line.find_first_of(",", start);
	}
	gb->palette.push_back(stof(line.substr(start, end - start).c_str()));

	gpu_palette_filter_init(gb);
}

void gpu_palette_filter_destroy(struct application_graph_node* agn) {
	struct gpu_palette_filter* mb = (struct gpu_palette_filter*)agn->component;

	cudaFree(mb->device_palette);
	delete mb;
}
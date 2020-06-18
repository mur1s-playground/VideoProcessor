#include "GPUVideoAlphaMerge.h"

#include "ApplicationGraph.h"
#include "MainUI.h"
#include "ComposeKernel.h"

#include <sstream>
#include <fstream>

void gpu_video_alpha_merge_init(struct gpu_video_alpha_merge* vam, int rgb_delay, int alpha_id) {
	vam->vs_rgb = nullptr;
	
	vam->rgb_delay = rgb_delay;
	vam->vs_alpha = nullptr;
	vam->channel_id = alpha_id;

	vam->vs_out = nullptr;
}

DWORD* gpu_video_alpha_merge_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agn->component;

	int last_frame_rgb = -1;
	int last_frame_alpha = -1;
	int current_out_frame = -1;
	while (agn->process_run) {
		gpu_memory_buffer_try_r(vam->vs_rgb->gmb, vam->vs_rgb->gmb->slots, true, 8);
		int next_frame_rgb = vam->vs_rgb->gmb->p_rw[2 * (vam->vs_rgb->gmb->slots + 1)];
		gpu_memory_buffer_release_r(vam->vs_rgb->gmb, vam->vs_rgb->gmb->slots);
		
		gpu_memory_buffer_try_r(vam->vs_alpha->gmb, vam->vs_alpha->gmb->slots, true, 8);
		int next_frame_alpha = vam->vs_alpha->gmb->p_rw[2 * (vam->vs_alpha->gmb->slots + 1)];
		gpu_memory_buffer_release_r(vam->vs_alpha->gmb, vam->vs_alpha->gmb->slots);

		current_out_frame = (current_out_frame + 1) % vam->vs_out->gmb->slots;
		
		if (next_frame_rgb != last_frame_rgb || next_frame_alpha != last_frame_alpha) {
			int next_frame_rgb_delayed = next_frame_rgb - vam->rgb_delay;
			if (next_frame_rgb_delayed < 0) next_frame_rgb_delayed += vam->vs_rgb->gmb->slots;

			gpu_memory_buffer_try_r(vam->vs_rgb->gmb, next_frame_rgb_delayed, true, 8);
			gpu_memory_buffer_try_r(vam->vs_alpha->gmb, next_frame_alpha, true, 8);
			gpu_memory_buffer_try_rw(vam->vs_out->gmb, current_out_frame, true, 8);
			
			compose_kernel_rgb_alpha_merge_launch(
				&vam->vs_rgb->gmb->p_device[next_frame_rgb_delayed * vam->vs_rgb->video_width * vam->vs_rgb->video_height* vam->vs_rgb->video_channels],
				&vam->vs_alpha->gmb->p_device[next_frame_alpha * vam->vs_alpha->video_width * vam->vs_alpha->video_height * vam->vs_alpha->video_channels],
				&vam->vs_out->gmb->p_device[current_out_frame * vam->vs_out->video_width * vam->vs_out->video_height * vam->vs_out->video_channels],
				vam->vs_out->video_width, vam->vs_out->video_height
			);

			gpu_memory_buffer_release_rw(vam->vs_out->gmb, current_out_frame);
			gpu_memory_buffer_release_r(vam->vs_alpha->gmb, next_frame_alpha);
			gpu_memory_buffer_release_r(vam->vs_rgb->gmb, next_frame_rgb_delayed);
			
			gpu_memory_buffer_try_rw(vam->vs_out->gmb, vam->vs_out->gmb->slots, true, 8);
			vam->vs_out->gmb->p_rw[2 * (vam->vs_out->gmb->slots + 1)] = current_out_frame;
			gpu_memory_buffer_release_rw(vam->vs_out->gmb, vam->vs_out->gmb->slots);
			
			last_frame_rgb = next_frame_rgb;
			last_frame_alpha = next_frame_alpha;
		} else {
			Sleep(16);
		}
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_video_alpha_merge_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agn->component;

	stringstream s_out;
	s_out << vam->rgb_delay << std::endl;
	s_out << vam->channel_id << std::endl;
	
	out_str = s_out.str();
}

void gpu_video_alpha_merge_load(struct gpu_video_alpha_merge* vam, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	int rgb_delay = stoi(line);
	std::getline(in_f, line);
	int alpha_id = stoi(line);
	
	gpu_video_alpha_merge_init(vam, rgb_delay, alpha_id);
}

void gpu_video_alpha_merge_destroy(struct application_graph_node* agn) {
	struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agn->component;
	delete vam;
}
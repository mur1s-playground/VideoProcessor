#include "GPUVideoAlphaMerge.h"

#include "ApplicationGraph.h"
#include "MainUI.h"
#include "ComposeKernel.h"

#include "Logger.h"

#include <sstream>
#include <fstream>

void gpu_video_alpha_merge_init(struct gpu_video_alpha_merge* vam, bool sync_prio_rgb, int alpha_id, int tps_target) {
	vam->vs_rgb = nullptr;
	
	vam->sync_prio_rgb = sync_prio_rgb;
	vam->vs_alpha = nullptr;
	vam->channel_id = alpha_id;
	vam->tps_target = tps_target;

	vam->vs_out = nullptr;
}

DWORD* gpu_video_alpha_merge_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agn->component;

	int last_frame_rgb = -1;
	unsigned long long last_frame_rgb_sync_time = 0;
	unsigned long long second_last_frame_rgb_sync_time = 0;

	int last_frame_alpha = -1;
	unsigned long long last_frame_alpha_sync_time = 0;
	unsigned long long second_last_frame_alpha_sync_time = 0;

	int current_out_frame = -1;
	unsigned long long sync_time = 0;
	unsigned long long last_sync_time = 0;

	agn->process_tps_balancer.tps_target = vam->tps_target;

	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		gpu_memory_buffer_try_r(vam->vs_rgb->gmb, vam->vs_rgb->gmb->slots, true, 8);
		int next_frame_rgb = vam->vs_rgb->gmb->p_rw[2 * (vam->vs_rgb->gmb->slots + 1)];
		gpu_memory_buffer_release_r(vam->vs_rgb->gmb, vam->vs_rgb->gmb->slots);
		
		gpu_memory_buffer_try_r(vam->vs_alpha->gmb, vam->vs_alpha->gmb->slots, true, 8);
		int next_frame_alpha = vam->vs_alpha->gmb->p_rw[2 * (vam->vs_alpha->gmb->slots + 1)];
		gpu_memory_buffer_release_r(vam->vs_alpha->gmb, vam->vs_alpha->gmb->slots);

		current_out_frame = (current_out_frame + 1) % vam->vs_out->gmb->slots;
		
		if (next_frame_rgb != last_frame_rgb || next_frame_alpha != last_frame_alpha) {
			unsigned long long sync_time = 0;
			int next_frame_rgb_id = next_frame_rgb;
			int next_frame_alpha_id = next_frame_alpha;
			if (vam->sync_prio_rgb) { //rgb prio
				gpu_memory_buffer_try_r(vam->vs_rgb->gmb, next_frame_rgb, true, 8);
				sync_time = gpu_memory_buffer_get_time(vam->vs_rgb->gmb, next_frame_rgb);
				
				unsigned long long candidate_sync_time = -1;
				int candidate_id = -1;

				for (int td = 0; td < vam->vs_alpha->gmb->slots; td++) {
					next_frame_alpha_id = next_frame_alpha - td;
					if (next_frame_alpha_id < 0) next_frame_alpha_id += vam->vs_alpha->gmb->slots;
					gpu_memory_buffer_try_r(vam->vs_alpha->gmb, next_frame_alpha_id, true, 8);
					unsigned long long tmp_sync_time = gpu_memory_buffer_get_time(vam->vs_alpha->gmb, next_frame_alpha_id);
					if (tmp_sync_time <= sync_time && tmp_sync_time > last_frame_alpha_sync_time && (tmp_sync_time > second_last_frame_rgb_sync_time || td == 0)) {
						candidate_id = next_frame_alpha_id;
						candidate_sync_time = tmp_sync_time;
					}
					gpu_memory_buffer_release_r(vam->vs_alpha->gmb, next_frame_alpha_id);
					if (tmp_sync_time <= last_frame_alpha_sync_time || tmp_sync_time <= second_last_frame_rgb_sync_time) {
						break;
					}
				}
				if (candidate_id > -1) {
					next_frame_alpha_id = candidate_id;
					last_frame_alpha_sync_time = candidate_sync_time;
				} else {
					next_frame_alpha_id = last_frame_alpha;
					if (next_frame_rgb_id == last_frame_rgb) {
						gpu_memory_buffer_release_r(vam->vs_rgb->gmb, next_frame_rgb);
						application_graph_tps_balancer_timer_stop(agn);
						application_graph_tps_balancer_sleep(agn);
						continue;
					}
				}
				if (last_frame_rgb_sync_time != sync_time) {
					second_last_frame_rgb_sync_time = last_frame_rgb_sync_time;
				}
				last_frame_rgb_sync_time = sync_time;
				gpu_memory_buffer_try_r(vam->vs_alpha->gmb, next_frame_alpha_id, true, 8);
			} else { //alpha prio
				gpu_memory_buffer_try_r(vam->vs_alpha->gmb, next_frame_alpha, true, 8);
				sync_time = gpu_memory_buffer_get_time(vam->vs_alpha->gmb, next_frame_alpha);

				unsigned long long candidate_sync_time = -1;
				int candidate_id = -1;

				for (int td = 0; td < vam->vs_rgb->gmb->slots; td++) {
					next_frame_rgb_id = next_frame_rgb - td;
					if (next_frame_rgb_id < 0) next_frame_rgb_id += vam->vs_rgb->gmb->slots;
					gpu_memory_buffer_try_r(vam->vs_rgb->gmb, next_frame_rgb_id, true, 8);
					unsigned long long tmp_sync_time = gpu_memory_buffer_get_time(vam->vs_rgb->gmb, next_frame_rgb_id);
					if (tmp_sync_time <= sync_time && tmp_sync_time > last_frame_rgb_sync_time && (tmp_sync_time > second_last_frame_alpha_sync_time || td == 0)) {
						candidate_id = next_frame_rgb_id;
						candidate_sync_time = tmp_sync_time;
					}
					gpu_memory_buffer_release_r(vam->vs_rgb->gmb, next_frame_rgb_id);
					if (tmp_sync_time <= last_frame_rgb_sync_time || tmp_sync_time <= second_last_frame_alpha_sync_time) {
						break;
					}
				}

				if (candidate_id > -1) {
					next_frame_rgb_id = candidate_id;
					last_frame_rgb_sync_time = candidate_sync_time;
				} else {
					next_frame_rgb_id = last_frame_rgb;
					if (next_frame_alpha_id == last_frame_alpha) {
						gpu_memory_buffer_release_r(vam->vs_alpha->gmb, next_frame_alpha);
						application_graph_tps_balancer_timer_stop(agn);
						application_graph_tps_balancer_sleep(agn);
						continue;
					}
				}
				if (last_frame_alpha_sync_time != sync_time) {
					second_last_frame_alpha_sync_time = last_frame_alpha_sync_time;
				}
				last_frame_alpha_sync_time = sync_time;
				gpu_memory_buffer_try_r(vam->vs_rgb->gmb, next_frame_rgb_id, true, 8);
			}

			gpu_memory_buffer_try_rw(vam->vs_out->gmb, current_out_frame, true, 8);
			
			compose_kernel_rgb_alpha_merge_launch(
				&vam->vs_rgb->gmb->p_device[next_frame_rgb_id * vam->vs_rgb->video_width * vam->vs_rgb->video_height* vam->vs_rgb->video_channels],
				&vam->vs_alpha->gmb->p_device[next_frame_alpha_id * vam->vs_alpha->video_width * vam->vs_alpha->video_height * vam->vs_alpha->video_channels], vam->vs_alpha->video_channels, vam->channel_id,
				&vam->vs_out->gmb->p_device[current_out_frame * vam->vs_out->video_width * vam->vs_out->video_height * vam->vs_out->video_channels],
				vam->vs_out->video_width, vam->vs_out->video_height
			);
			gpu_memory_buffer_set_time(vam->vs_out->gmb, current_out_frame, sync_time);

			gpu_memory_buffer_release_rw(vam->vs_out->gmb, current_out_frame);
			gpu_memory_buffer_release_r(vam->vs_alpha->gmb, next_frame_alpha_id);
			gpu_memory_buffer_release_r(vam->vs_rgb->gmb, next_frame_rgb_id);
			
			gpu_memory_buffer_try_rw(vam->vs_out->gmb, vam->vs_out->gmb->slots, true, 8);
			vam->vs_out->gmb->p_rw[2 * (vam->vs_out->gmb->slots + 1)] = current_out_frame;
			gpu_memory_buffer_release_rw(vam->vs_out->gmb, vam->vs_out->gmb->slots);
			
			last_frame_rgb = next_frame_rgb_id;
			last_frame_alpha = next_frame_alpha_id;
		}
		
		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_video_alpha_merge_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agn->component;

	stringstream s_out;
	if (vam->sync_prio_rgb) {
		s_out << 1 << std::endl;
	} else {
		s_out << 0 << std::endl;
	}
	s_out << vam->channel_id << std::endl;
	s_out << vam->tps_target << std::endl;
	
	out_str = s_out.str();
}

void gpu_video_alpha_merge_load(struct gpu_video_alpha_merge* vam, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	bool sync_prio_rgb = (stoi(line) == 1);
	std::getline(in_f, line);
	int alpha_id = stoi(line);
	std::getline(in_f, line);
	int tps = stoi(line);
	
	gpu_video_alpha_merge_init(vam, sync_prio_rgb, alpha_id, tps);
	vam->tps_target = tps;
}

void gpu_video_alpha_merge_destroy(struct application_graph_node* agn) {
	struct gpu_video_alpha_merge* vam = (struct gpu_video_alpha_merge*)agn->component;
	delete vam;
}
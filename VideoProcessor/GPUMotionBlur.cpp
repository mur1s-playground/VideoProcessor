#include "GPUMotionBlur.h"

#include "MotionBlur.h"

#include "ApplicationGraph.h"

#include <sstream>
#include <fstream>

#include "Logger.h"

#include "MainUI.h"

void gpu_motion_blur_init(struct gpu_motion_blur* mb, int frame_count, int weight_dist_type, float frame_id_weight_center, float center_weight) {
	mb->frame_count = frame_count;
	mb->weight_dist_type = weight_dist_type;
	mb->frame_id_weight_center = frame_id_weight_center;
	mb->c = center_weight;
	mb->vs_in = nullptr;
	mb->gmb_out = nullptr;
}

void gpu_motion_blur_calculate_weights(struct gpu_motion_blur* mb) {
	mb->calc_err = false;
	if (mb->weight_dist_type == 0) { // even
		mb->a = 0.0f;
		mb->b = 0.0f;
		mb->c = 1.0f / mb->frame_count;
	} else if (mb->weight_dist_type == 1) { // linear roof
		float value = 0.0f;
		float err = 1.0f;
		mb->a = 0.0f;
		mb->b = 0.0f;
		while (abs(err) > 0.01f && !mb->calc_err) {
			value = 0.0f;
			for (int i = 0; i < mb->frame_count; i++) {
				float tmp_val = mb->c;
				if (i < mb->frame_id_weight_center) {
					tmp_val += mb->a * (i - mb->frame_id_weight_center);
				} else if (i > mb->frame_id_weight_center) {
					tmp_val += mb->b * (i - mb->frame_id_weight_center);
				}
				if (tmp_val < 0) {
					mb->calc_err = true;
					break;
				}
				value += tmp_val;
			}
			err = 1.0f - value;
			if (mb->frame_id_weight_center == 0) {
				mb->a = 0.0f;
				if (err > 0.01) {
					mb->b += 0.001;
				} else if (err < -0.01) {
					mb->b -= 0.001;
				}
			} else if (mb->frame_id_weight_center == mb->frame_count - 1) {
				mb->b = 0.0f;
				if (err > 0.01) {
					mb->a -= 0.001;
				} else if (err < -0.01) {
					mb->a += 0.001;
				}
			} else {
				if (err > 0.01) {
					mb->a -= 0.001;
				} else if (err < -0.01) {
					mb->a += 0.001;
				}
				mb->b = -mb->a * (mb->frame_count - 1 - mb->frame_id_weight_center) / (mb->frame_id_weight_center);
			}
		}
	}
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

				motion_blur_kernel_launch(&mb->vs_in->gmb->p_device[id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels], &mb->gmb_out->p_device[next_gpu_out_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels], mb->vs_in->video_width, mb->vs_in->video_height, mb->vs_in->video_channels, mb->frame_count, frame_counter, mb->weight_dist_type, mb->frame_id_weight_center, mb->a, mb->b, mb->c, true);

				if (i == next_gpu_id) {
					gpu_memory_buffer_set_time(mb->gmb_out, next_gpu_out_id, gpu_memory_buffer_get_time(mb->vs_in->gmb, id));
				}

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
	s_out << mb->weight_dist_type << std::endl;
	s_out << mb->frame_id_weight_center << std::endl;
	s_out << mb->c << std::endl;
	
	out_str = s_out.str();
}

void gpu_motion_blur_load(struct gpu_motion_blur* mb, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	mb->frame_count = stoi(line);
	std::getline(in_f, line);
	mb->weight_dist_type = stoi(line);
	std::getline(in_f, line);
	mb->frame_id_weight_center = stof(line);
	std::getline(in_f, line);
	mb->c = stof(line);
	
	mb->vs_in = nullptr;
	mb->gmb_out = nullptr;
	gpu_motion_blur_calculate_weights(mb);
}

void gpu_motion_blur_destroy(struct application_graph_node* agn) {
	struct gpu_motion_blur* mb = (struct gpu_motion_blur*)agn->component;
	delete mb;
}

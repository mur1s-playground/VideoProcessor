#include "Statistics3D.h"

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "CameraControl.h"

#include "StatisticsKernel.h"

#include "Logger.h"

void statistics_3d_init(struct statistics_3d *s3d) {
	s3d->smb_shared_state = nullptr;

	s3d->vs_out = nullptr;
}

DWORD* statistics_3d_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct statistics_3d* s3d = (struct statistics_3d*)agn->component;

	int last_state_slot = -1;

	int slots_per_slot = s3d->smb_shared_state->size / sizeof(struct camera_control_shared_state);

	struct camera_control_shared_state* ccss_local = (struct camera_control_shared_state*)malloc(slots_per_slot * 2 * sizeof(struct camera_control_shared_state));
	memset(ccss_local, 0, slots_per_slot * 2 * sizeof(struct camera_control_shared_state));

	bool local_switch = false;
	struct camera_control_shared_state* ccss_local_current = &ccss_local[local_switch * slots_per_slot * sizeof(struct camera_control_shared_state)];
	struct camera_control_shared_state* ccss_local_last = &ccss_local[!local_switch * slots_per_slot * sizeof(struct camera_control_shared_state)];

	struct vector3<float>* velocity_buffer = (struct vector3<float> *) malloc(slots_per_slot * 5 * sizeof(struct vector3<float>));
	memset(velocity_buffer, 0, slots_per_slot * 5 * sizeof(struct vector3<float>));
	
	struct camera_control_shared_state* ccss;

	//TMP NO CFG
	statistic_heatmap_init(&s3d->heatmap_3d, struct vector2<int>(-5, 25), struct vector2<int>(-5, 25), struct vector2<int>(-1, 1), struct vector3<float>(1.0f, 1.0f, 1.0f), 0.995f);
	statistic_vectorfield_3d_init(&s3d->movement_vectorfield_3d, struct vector2<int>(-5, 25), struct vector2<int>(-5, 25), struct vector2<int>(-1, 1), struct vector3<float>(1.0f, 1.0f, 1.0f), 10);
	//

	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		shared_memory_buffer_try_r(s3d->smb_shared_state, s3d->smb_shared_state->slots, true, 8);
		int current_state_slot = s3d->smb_shared_state->p_buf_c[s3d->smb_shared_state->slots * s3d->smb_shared_state->size + ((s3d->smb_shared_state->slots + 1) * 2)];
		shared_memory_buffer_release_r(s3d->smb_shared_state, s3d->smb_shared_state->slots);

		if (current_state_slot != last_state_slot) {
			shared_memory_buffer_try_r(s3d->smb_shared_state, current_state_slot, true, 8);
			ccss = (struct camera_control_shared_state*)&s3d->smb_shared_state->p_buf_c[current_state_slot * s3d->smb_shared_state->size];
			memcpy(&ccss_local[local_switch * slots_per_slot], ccss, slots_per_slot * sizeof(struct camera_control_shared_state));
			shared_memory_buffer_release_r(s3d->smb_shared_state, current_state_slot);
			ccss_local_current = &ccss_local[local_switch * slots_per_slot];

			unsigned long long timest = 0;

			for (int ca = 0; ca < slots_per_slot; ca++) {
				for (int d = 0; d < 5; d++) {
					struct cam_detection_3d* last_detection = &ccss_local_last[ca].latest_detections_3d[d];
					struct cam_detection_3d* current_detection = &ccss_local_current[ca].latest_detections_3d[d];

					const float timestep = 400000;

					if (current_detection->timestamp != 0 && last_detection->timestamp != 0) {
						timest = current_detection->timestamp;
						struct vector3<float> velocity = current_detection->position - last_detection->position;
						float timediff = (float)(current_detection->timestamp - last_detection->timestamp);
						//logger(timediff);
						//timediff /= timestep;
						
						//TMP if object replaced, tp threshold
						if (length(velocity) < 5.0f) {
							struct vector3<float> acceleration = (velocity/timediff - velocity_buffer[ca * 5 + d])/timediff;
							velocity_buffer[ca * 5 + d] = velocity/timediff;
							//logger("velocity", length(velocity / timediff));
							//logger("acceleration", length(acceleration));
							statistic_vectorfield_3d_update(&s3d->movement_vectorfield_3d, current_detection->position, velocity, length(velocity/timediff), length(acceleration));
							statistic_heatmap_update(&s3d->heatmap_3d, current_detection->position);
						}
					}
				}
			}

			statistic_heatmap_update_calculate(&s3d->heatmap_3d);
			if (s3d->vs_out != nullptr) {
				statistic_vectorfield_3d_update_device(&s3d->movement_vectorfield_3d);

				gpu_memory_buffer_try_r(s3d->vs_out->gmb, s3d->vs_out->gmb->slots, true, 8);
				int next_gpu_id = (s3d->vs_out->gmb->p_rw[2 * (s3d->vs_out->gmb->slots + 1)] + 1) % s3d->vs_out->gmb->slots;
				gpu_memory_buffer_release_r(s3d->vs_out->gmb, s3d->vs_out->gmb->slots);
				gpu_memory_buffer_try_rw(s3d->vs_out->gmb, next_gpu_id, true, 8);
				
				statistics_3d_kernel_launch(s3d->heatmap_3d.device_data, s3d->movement_vectorfield_3d.device_data, s3d->movement_vectorfield_3d.max_vel, s3d->movement_vectorfield_3d.max_acc, s3d->vs_out->gmb->p_device + (next_gpu_id * s3d->vs_out->gmb->size), s3d->vs_out->video_width, s3d->vs_out->video_height, struct vector3<int>(s3d->heatmap_3d.sqg.dimensions[0], s3d->heatmap_3d.sqg.dimensions[1], s3d->heatmap_3d.sqg.dimensions[2]));
				
				gpu_memory_buffer_set_time(s3d->vs_out->gmb, next_gpu_id, timest);
				gpu_memory_buffer_release_rw(s3d->vs_out->gmb, next_gpu_id);
				gpu_memory_buffer_try_rw(s3d->vs_out->gmb, s3d->vs_out->gmb->slots, true, 8);
				s3d->vs_out->gmb->p_rw[2 * (s3d->vs_out->gmb->slots + 1)] = next_gpu_id;
				gpu_memory_buffer_release_rw(s3d->vs_out->gmb, s3d->vs_out->gmb->slots);
			}

			ccss_local_last = ccss_local_current;
			local_switch = !local_switch;
			last_state_slot = current_state_slot;
		}

		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}


void statistics_3d_externalise(struct application_graph_node* agn, string& out_str) {
	struct camera_control* cc = (struct camera_control*)agn->component;

	stringstream s_out;
	out_str = s_out.str();
}

void statistics_3d_load(struct statistics_3d* s3d, ifstream& in_f) {
	std::string line;

	statistics_3d_init(s3d);
}

void statistics_3d_destroy(struct application_graph_node* agn) {

}
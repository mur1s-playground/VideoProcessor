#include "DetectionSimulation3D.h"

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "CameraControl.h"

void detection_simulation_3d_set_example_cfg(struct detection_simulation_3d* ds3d);

void detection_simulation_3d_init(struct detection_simulation_3d* ds3d) {
	//TMP NO CFG
	detection_simulation_3d_set_example_cfg(ds3d);
	//-----//
	ds3d->smb_size_req = ds3d->object_count * sizeof(struct cam_detection_3d);

	ds3d->smb_detections = nullptr;
}

DWORD* detection_simulation_3d_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct  detection_simulation_3d* ds3d = (struct detection_simulation_3d*)agn->component;

	std::vector<struct cam_detection_3d> cd3d;
	struct cam_detection_3d cdtmp;
	memset(&cdtmp, 0, sizeof(struct cam_detection_3d));

	int tick = 0;
	int current_smb_id = -1;

	float *cam_dists = (float *) malloc(ds3d->object_count * sizeof(float));
	int *cam_sort_idx = (int*)malloc(ds3d->object_count * sizeof(int));

	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		for (int o = 0; o < ds3d->object_count; o++) {
			bool active_changed = false;

			if (cd3d.size() < o + 1) {
				cd3d.push_back(cdtmp);
			}

			if (tick % ds3d->tick_speed == 0) {
				ds3d->keypoint_active[o] = ds3d->keypoint_next[o];
				cd3d[o].position = ds3d->keypoints[ds3d->keypoint_active[o]];
				active_changed = true;
			}

			std::vector<int> current_keypoint_candidates = ds3d->flowmap[ds3d->keypoint_active[o]];

			//change or update target keypoint
			float prob_change = (int)roundf((rand() / (float)RAND_MAX));

			if ((prob_change < ds3d->probability_of_change && tick % ds3d->tick_speed < 0.5 * ds3d->tick_speed) || active_changed) {
				int keypoint_id = (int)floorf((rand() / (float)RAND_MAX) * current_keypoint_candidates.size());
				if (keypoint_id == current_keypoint_candidates.size()) keypoint_id--;
				//cout << "size " << current_keypoint_candidates.size() << " id" << keypoint_id << std::endl;
				//cout << keypoint_id << std::endl;
				ds3d->keypoint_next[o] = current_keypoint_candidates[keypoint_id];
			}

			float len_t = length(ds3d->keypoints[ds3d->keypoint_next[o]] - ds3d->keypoints[ds3d->keypoint_active[o]]);

			float len_2t = length(ds3d->keypoints[ds3d->keypoint_next[o]] - cd3d[o].position) + 1e-9;
			struct vector3<float> direction = (ds3d->keypoints[ds3d->keypoint_next[o]] - cd3d[o].position) * (1.0f / (float)len_2t);


			float dist = (1.0f - ((tick % ds3d->tick_speed) / (float)ds3d->tick_speed)) * len_t;

			cd3d[o].velocity = cd3d[o].position;

			cd3d[o].position = cd3d[o].position - direction * (-1.0f / (float)ds3d->tick_speed * len_t) * (len_2t / dist);

			cd3d[o].velocity = cd3d[o].position - cd3d[o].velocity;

			//cout << o << " " <<  cd3d[o].position[0] << " " << cd3d[o].position[1] << " " << cd3d[o].position[2] << std::endl;

			//cout << tick << " " << o << std::endl;

			int	cam_sort = 0;

			for (int ca = 0; ca < ds3d->camera_positions.size(); ca++) {
				cam_dists[ca] = length(cd3d[o].position - ds3d->camera_positions[ca]);
			}
			while (cam_sort < ds3d->camera_positions.size()) {
				float cam_min_dist = 100000.0f;
				int cam_min_idx = -1;

				for (int ca = 0; ca < ds3d->camera_positions.size(); ca++) {
					if (cam_dists[ca] < cam_min_dist) {
						bool found = false;
						for (int cc = 0; cc < cam_sort; cc++) {
							if (cam_sort_idx[cc] == ca) {
								found = true;
								break;
							}
						}
						if (!found) {
							cam_min_dist = cam_dists[ca];
							cam_min_idx = ca;
						}
					}
				}
				cam_sort_idx[cam_sort] = cam_min_idx;
				cam_sort++;
			}
			for (int r = 0; r < 3; r++) {
				cd3d[o].ray_position[r] = ds3d->camera_positions[cam_sort_idx[r]];
				cd3d[o].ray_direction[r] = cd3d[o].position - cd3d[o].ray_position[r];
			}

			cd3d[o].timestamp = tick;
			cd3d[o].class_id = 37;

			cd3d[o].score = 0.5;
		}

		int current_id = (current_smb_id + 1) % ds3d->smb_detections->slots;
		shared_memory_buffer_try_rw(ds3d->smb_detections, current_id, true, 8);
		memcpy(&ds3d->smb_detections->p_buf_c[current_id * ds3d->object_count * sizeof(struct cam_detection_3d)], cd3d.data(), ds3d->object_count * sizeof(struct cam_detection_3d));
		shared_memory_buffer_release_rw(ds3d->smb_detections, current_id);

		shared_memory_buffer_try_rw(ds3d->smb_detections, ds3d->smb_detections->slots, true, 8);
		ds3d->smb_detections->p_buf_c[ds3d->smb_detections->slots * ds3d->object_count * sizeof(struct cam_detection_3d) + (ds3d->smb_detections->slots + 1) * 2] = current_id;
		shared_memory_buffer_release_rw(ds3d->smb_detections, ds3d->smb_detections->slots);

		current_smb_id = current_id;

		tick++;

		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void detection_simulation_3d_externalise(struct application_graph_node* agn, string& out_str) {
	struct detection_simulation_3d* ds3d = (struct detection_simulation_3d*)agn->component;

	stringstream s_out;
	out_str = s_out.str();
}

void detection_simulation_3d_load(struct detection_simulation_3d* ds3d, ifstream& in_f) {
	std::string line;

	detection_simulation_3d_init(ds3d);
}

void detection_simulation_3d_destroy(struct application_graph_node* agn) {

}

void detection_simulation_3d_set_example_cfg(struct detection_simulation_3d* ds3d) {
	ds3d->camera_positions.push_back({ 25.0, 0.0f, 10.0f });
	ds3d->camera_positions.push_back({ 25.0, 16.0f, 10.0f });
	ds3d->camera_positions.push_back({ 10.0, 25.0f, 10.0f });
	ds3d->camera_positions.push_back({ -4.0, 16.0f, 10.0f });
	ds3d->camera_positions.push_back({ -4.0, 1.0f, 10.0f });

	ds3d->keypoints.push_back({ 18.0f,	1.0f,	0.0f });
	ds3d->keypoints.push_back({ 18.0f,	4.0f,	0.0f });

	ds3d->keypoints.push_back({ 16.0f,	7.0f,	0.0f });
	ds3d->keypoints.push_back({ 18.0f,	7.0f,	0.0f });
	ds3d->keypoints.push_back({ 20.0f,	7.0f,	0.0f });

	ds3d->keypoints.push_back({ 16.0f,	10.0f,	0.0f });
	ds3d->keypoints.push_back({ 18.0f,	10.0f,	0.0f });
	ds3d->keypoints.push_back({ 20.0f,	10.0f,	0.0f });

	ds3d->keypoints.push_back({ 16.0f,	13.0f,	0.0f });
	ds3d->keypoints.push_back({ 18.0f,	13.0f,	0.0f });
	ds3d->keypoints.push_back({ 20.0f,	13.0f,	0.0f });

	ds3d->keypoints.push_back({ 18.0f,	16.0f,	0.0f });

	ds3d->keypoints.push_back({ 18.0f,	19.0f,	0.0f });

	ds3d->keypoints.push_back({ 10.0f,	22.0f,	0.0f });

	ds3d->keypoints.push_back({ 4.0f,	19.0f,	0.0f });

	ds3d->keypoints.push_back({ 2.0f,	16.0f,	0.0f });
	ds3d->keypoints.push_back({ 4.0f,	16.0f,	0.0f });
	ds3d->keypoints.push_back({ 6.0f,	16.0f,	0.0f });

	ds3d->keypoints.push_back({ 2.0f,	13.0f,	0.0f });
	ds3d->keypoints.push_back({ 4.0f,	13.0f,	0.0f });
	ds3d->keypoints.push_back({ 6.0f,	13.0f,	0.0f });

	ds3d->keypoints.push_back({ 2.0f,	10.0f,	0.0f });
	ds3d->keypoints.push_back({ 4.0f,	10.0f,	0.0f });
	ds3d->keypoints.push_back({ 6.0f,	10.0f,	0.0f });

	ds3d->keypoints.push_back({ 2.0f,	7.0f,	0.0f });
	ds3d->keypoints.push_back({ 4.0f,	7.0f,	0.0f });
	ds3d->keypoints.push_back({ 6.0f,	7.0f,	0.0f });

	ds3d->keypoints.push_back({ 4.0f,	4.0f,	0.0f });

	ds3d->keypoints.push_back({ 6.0f,	1.0f,	0.0f });


	ds3d->flowmap[0] = std::vector<int>{ 1 };
	ds3d->flowmap[1] = std::vector<int>{ 2, 3, 4 };

	ds3d->flowmap[2] = std::vector<int>{ 5, 6, 7 };
	ds3d->flowmap[3] = std::vector<int>{ 5, 6, 7 };
	ds3d->flowmap[4] = std::vector<int>{ 5, 6, 7 };

	ds3d->flowmap[5] = std::vector<int>{ 8, 9, 10 };
	ds3d->flowmap[6] = std::vector<int>{ 8, 9, 10 };
	ds3d->flowmap[7] = std::vector<int>{ 8, 9, 10 };

	ds3d->flowmap[8] = std::vector<int>{ 11 };
	ds3d->flowmap[9] = std::vector<int>{ 11 };
	ds3d->flowmap[10] = std::vector<int>{ 11 };

	ds3d->flowmap[11] = std::vector<int>{ 12 };

	ds3d->flowmap[12] = std::vector<int>{ 13 };

	ds3d->flowmap[13] = std::vector<int>{ 14 };

	ds3d->flowmap[14] = std::vector<int>{ 15, 16, 17 };

	ds3d->flowmap[15] = std::vector<int>{ 18, 19, 20 };
	ds3d->flowmap[16] = std::vector<int>{ 18, 19, 20 };
	ds3d->flowmap[17] = std::vector<int>{ 18, 19, 20 };

	ds3d->flowmap[18] = std::vector<int>{ 21, 22, 23 };
	ds3d->flowmap[19] = std::vector<int>{ 21, 22, 23 };
	ds3d->flowmap[20] = std::vector<int>{ 21, 22, 23 };

	ds3d->flowmap[21] = std::vector<int>{ 24, 25, 26 };
	ds3d->flowmap[22] = std::vector<int>{ 24, 25, 26 };
	ds3d->flowmap[23] = std::vector<int>{ 24, 25, 26 };

	ds3d->flowmap[24] = std::vector<int>{ 27 };
	ds3d->flowmap[25] = std::vector<int>{ 27 };
	ds3d->flowmap[26] = std::vector<int>{ 27 };

	ds3d->flowmap[27] = std::vector<int>{ 28 };

	ds3d->flowmap[28] = std::vector<int>{ 0 };


	ds3d->tick_speed = 30;
	ds3d->detected_class_id = 37;
	ds3d->object_count = 7;
	ds3d->probability_of_change = 0.0025f;

	int flowmap_size = ds3d->flowmap.size();
	for (int o = 0; o < ds3d->object_count; o++) {
		int idx = (int)(((flowmap_size-1) / (float)(ds3d->object_count)) * o);
		int rnd = (int)floorf((rand() / (float)RAND_MAX) * (ds3d->flowmap[idx].size()-1));
		if (rnd > ds3d->flowmap[idx].size() - 1) rnd--;
		ds3d->keypoint_active.push_back(ds3d->flowmap[idx][rnd]);
		ds3d->keypoint_next.push_back(ds3d->flowmap[idx][rnd]);
	}
}
#include "Statistic.h"

#include <math.h>
#include <cstdlib>
#include "Vector2.h"
#include "Logger.h"

void statistic_angle_denoiser_init(struct statistic_angle_denoiser* sad, int size) {
	sad->angle = 0.0f;
	sad->angle_stability = 1.0f;
	sad->angle_distribution_size = size;
	sad->angle_distribution_idx_latest = -1;
	sad->angle_distribution = (float*)malloc(sad->angle_distribution_size * sizeof(float));
	sad->angle_distribution_weights = (float*)malloc(sad->angle_distribution_size * sizeof(float));
	for (int adw = 0; adw < sad->angle_distribution_size; adw++) {
		sad->angle_distribution_weights[adw] = 1.0f / (float)sad->angle_distribution_size;
		sad->angle_distribution[adw] = 0.0f;
	}
}

void statistic_angle_denoiser_set_weights(struct statistic_angle_denoiser* sad, float* weights) {
	for (int adw = 0; adw < sad->angle_distribution_size; adw++) {
		sad->angle_distribution_weights[adw] = weights[adw];
	}
}

bool statistic_angle_denoiser_is_left_of(float angle_base, float angle_new) {
	if (angle_base >= 0.0f && angle_base < 90.0f && angle_new <= 360.0f && angle_new > 270.0f) {
		return true;
	} else if (angle_new >= 0.0f && angle_new < 90.0f && angle_base <= 360.0f && angle_base > 270.0f) {
		return false;
	}
	return (angle_base - angle_new > 0);
}

void statistic_angle_denoiser_update(struct statistic_angle_denoiser* sad, float angle) {
	sad->angle_distribution_idx_latest = (sad->angle_distribution_idx_latest + 1) % sad->angle_distribution_size;
	sad->angle_distribution[sad->angle_distribution_idx_latest] = angle;
	if (sad->angle_distribution[sad->angle_distribution_idx_latest] < 0) sad->angle_distribution[sad->angle_distribution_idx_latest] += 360.0f;
	float np_tmp = (float)sad->angle_distribution[sad->angle_distribution_idx_latest];
	sad->angle_stability = 1.0f;
	int tmp_d_c = sad->angle_distribution_idx_latest;
	for (int np_d = 0; np_d < sad->angle_distribution_size; np_d++) {
		float f_dt = 0.0f;
		float dt_s = 1 - (2 * (statistic_angle_denoiser_is_left_of(sad->angle_distribution[sad->angle_distribution_idx_latest], sad->angle_distribution[tmp_d_c])));
		float dt = (float)abs(sad->angle_distribution[sad->angle_distribution_idx_latest] - sad->angle_distribution[tmp_d_c]);
		if (dt < 180.0f) {
			f_dt = dt;
		} else {
			f_dt = 360.0f - dt;
		}
		np_tmp += (sad->angle_distribution_weights[np_d] * dt_s * f_dt);
		f_dt /= 180.0f;
		sad->angle_stability -= f_dt * sad->angle_distribution_weights[np_d];
		tmp_d_c = (tmp_d_c + 1) % sad->angle_distribution_size;
	}
	if (np_tmp < 0) np_tmp += 360;
	if (np_tmp > 360) np_tmp -= 360;
	sad->angle = np_tmp;
}

void statistic_detection_matcher_2d_init(struct statistic_detection_matcher_2d* sdm2, int size, unsigned long long ttl, int avg_size) {
	sdm2->size = size;
	sdm2->detections = (struct cam_detection*)malloc(sdm2->size*sizeof(cam_detection));
	memset(sdm2->detections, 0, size * sizeof(cam_detection));
	sdm2->matches_history = (struct cam_detection_history*) malloc(sdm2->size * sizeof(struct cam_detection_history));
	for (int mh = 0; mh < sdm2->size; mh++) {
		sdm2->matches_history[mh].size = avg_size;
		sdm2->matches_history[mh].latest_idx = -1;
		sdm2->matches_history[mh].latest_count = 0;
		sdm2->matches_history[mh].history = (struct cam_detection*)malloc(sdm2->matches_history[mh].size * sizeof(struct cam_detection));
		memset(sdm2->matches_history[mh].history, 0, sdm2->matches_history[mh].size * sizeof(struct cam_detection));
	}
	sdm2->ttl = ttl;
}

void statistic_detection_matcher_2d_update(struct statistic_detection_matcher_2d* sdm2, struct cam_detection_history* cdh) {
	for (int c = 0; c < cdh->latest_count; c++) {
		int cur_h_idx = cdh->latest_idx - c;
		if (cur_h_idx < 0) cur_h_idx += cdh->size;
		struct cam_detection* current_detection = &cdh->history[cur_h_idx];

		struct vector2<float> center = { current_detection->x1 + 0.5f * (current_detection->x2 - current_detection->x1), current_detection->y1 + 0.5f * (current_detection->y2 - current_detection->y1) };
		int width = current_detection->x2 - current_detection->x1;
		int height = current_detection->y2 - current_detection->y1;

		int best_idx = -1;
		float best_score = 0.0f;
		int free_tracker_idx = -1;

		for (int s = 0; s < sdm2->size; s++) {
			if (sdm2->detections[s].timestamp > 0) {
				if (current_detection->timestamp - sdm2->detections[s].timestamp > sdm2->ttl) {
					sdm2->detections[s].timestamp = 0;
					sdm2->matches_history[s].latest_idx = -1;
					sdm2->matches_history[s].latest_count = 0;
					memset(sdm2->matches_history[s].history, 0, sdm2->matches_history[s].size * sizeof(struct cam_detection));
					if (free_tracker_idx < 0) {
						free_tracker_idx = s;
					}
				} else {
					if (sdm2->detections[s].class_id != current_detection->class_id) {
						continue;
					}
					struct vector2<float> center_s = { sdm2->detections[s].x1 + 0.5f * (sdm2->detections[s].x2 - sdm2->detections[s].x1), sdm2->detections[s].y1 + 0.5f * (sdm2->detections[s].y2 - sdm2->detections[s].y1) };
					float center_d = length(center_s - center);
					float center_sc = center_d / width;
					if (center_sc > 1.0f) {
						continue;
					}

					int width_s = sdm2->detections[s].x2 - sdm2->detections[s].x1;
					float width_r = width / width_s;
					float width_r_sc = abs(1 - width_r);
					if (width_r_sc > 0.2) {
						continue;
					}

					int height_s = sdm2->detections[s].y2 - sdm2->detections[s].y1;
					float height_r = height / height_s;
					float height_r_sc = abs(1 - height_r);
					if (height_r_sc > 0.2) {
						continue;
					}

					float score_tmp = 1.0f - center_sc - width_r_sc - height_r_sc;
					if (score_tmp > 0) {
						if (score_tmp > best_score) {
							best_score = score_tmp;
							best_idx = s;
						}
					}
				}
			} else {
				if (free_tracker_idx < 0) {
					free_tracker_idx = s;
				}
			}
		}
		if (best_idx > -1) {
			sdm2->matches_history[best_idx].latest_idx = (sdm2->matches_history[best_idx].latest_idx + 1) % sdm2->matches_history[best_idx].size;
			memcpy(&sdm2->matches_history[best_idx].history[sdm2->matches_history[best_idx].latest_idx], current_detection, sizeof(struct cam_detection));
			sdm2->detections[best_idx].timestamp = current_detection->timestamp;
			if (sdm2->matches_history[best_idx].latest_count < sdm2->matches_history[best_idx].size) {
				sdm2->matches_history[best_idx].latest_count++;
			}
			float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;
			int included_total = 0;
			for (int av = 0; av < sdm2->matches_history[best_idx].size; av++) {
				x1 += sdm2->matches_history[best_idx].history[av].x1;
				x2 += sdm2->matches_history[best_idx].history[av].x2;
				y1 += sdm2->matches_history[best_idx].history[av].y1;
				y2 += sdm2->matches_history[best_idx].history[av].y2;
			}
			sdm2->detections[best_idx].x1 = (int)(x1 / (float)sdm2->matches_history[best_idx].latest_count);
			sdm2->detections[best_idx].x2 = (int)(x2 / (float)sdm2->matches_history[best_idx].latest_count);
			sdm2->detections[best_idx].y1 = (int)(y1 / (float)sdm2->matches_history[best_idx].latest_count);
			sdm2->detections[best_idx].y2 = (int)(y2 / (float)sdm2->matches_history[best_idx].latest_count);
		} else {
			if (free_tracker_idx > -1) {
				memcpy(&sdm2->detections[free_tracker_idx], current_detection, sizeof(struct cam_detection));
				memcpy(&sdm2->matches_history[free_tracker_idx].history[0], current_detection, sizeof(struct cam_detection));
				sdm2->matches_history[free_tracker_idx].latest_idx = 0;
				sdm2->matches_history[free_tracker_idx].latest_count++;
			}
		}
	}
}

int statistic_detection_matcher_2d_get_stable_match(struct statistic_detection_matcher_2d* sdm2, int class_id, int count_threshold) {
	for (int cd = 0; cd < sdm2->size; cd++) {
		if (sdm2->detections[cd].class_id == class_id) {
			if (sdm2->matches_history[cd].latest_count == count_threshold) {
				return cd;
			}
		}
	}
	return -1;
}

void statistic_detection_matcher_3d_init(struct statistic_detection_matcher_3d* sdm3, int size, unsigned long long ttl, struct camera_control* cc) {
	sdm3->detections = (struct cam_detection_3d*)malloc(size*sizeof(struct cam_detection_3d));
	memset(sdm3->detections, 0, size * sizeof(struct cam_detection_3d));
	sdm3->is_final_matched = (bool*)malloc(size * sizeof(bool));
	memset(sdm3->is_final_matched, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(bool));

	sdm3->size = size;
	sdm3->ttl = ttl;

	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		if (cdh->size > sdm3->cdh_max_size) {
			sdm3->cdh_max_size = cdh->size;
		}
	}

	sdm3->is_matched = (bool*)malloc(cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
	sdm3->detections_buffer = (struct cam_detection_3d*)malloc(cc->camera_count * sdm3->cdh_max_size * sizeof(struct cam_detection_3d));
	sdm3->detections_3d = (struct statistic_detection_matcher_3d_detection*)malloc(cc->camera_count * sdm3->cdh_max_size * sizeof(struct statistic_detection_matcher_3d_detection));

	sdm3->class_match_matrix = (bool *)malloc(cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
	sdm3->distance_matrix = (float*)malloc(cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float));
	sdm3->min_dist_central_points_matrix = (struct vector3<float> *) malloc(cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(struct vector3<float>));
	sdm3->size_factor_matrix = (float *)malloc(cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float));
}

void statistic_detection_matcher_3d_update(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc) {
	float distance_unit = cc->cam_awareness[0].calibration.d_1;

	//precompute ray meta
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		for (int c = 0; c < cdh->latest_count; c++) {
			int cur_h_idx = cdh->latest_idx - c;
			if (cur_h_idx < 0) cur_h_idx += cdh->size;
			struct cam_detection* current_detection = &cdh->history[cur_h_idx];

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].class_id = current_detection->class_id;

			struct vector2<float> det_center = cam_detection_get_center(current_detection);

			float north_pole = cc->cam_awareness[ca].north_pole.angle;
			float horizon = cc->cam_awareness[ca].horizon.angle;

			struct vector2<int> diff_from_mid = {
				(int)(cc->cam_meta[ca].resolution[0] / 2.0f - det_center[0]),
				(int)(cc->cam_meta[ca].resolution[1] / 2.0f - det_center[1])
			};

			float fov_np = cc->cam_awareness[ca].calibration.lens_fov[0];
			float fov_h = cc->cam_awareness[ca].calibration.lens_fov[1];

			//TODO: shift angles_vec by probably 90 degree
			struct vector2<float> angles_vec = {
				north_pole - (diff_from_mid[0] / (cc->cam_meta[ca].resolution[0] / 2.0f)) * (fov_np / 2.0f) + 90.0f,
				horizon - (diff_from_mid[1] / (cc->cam_meta[ca].resolution[1] / 2.0f)) * (fov_h / 2.0f) + 90.0f
			};

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction = {
				sinf(angles_vec[1]) * cosf(angles_vec[0]),
				sinf(angles_vec[1]) * sinf(angles_vec[0]),
				cosf(angles_vec[1])
			};

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].dimensions = {
				current_detection->x2 - current_detection->x1,
				current_detection->y2 - current_detection->y1
			};

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].timestamp = current_detection->timestamp;
		}
	}

	//precompute ray correlation
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		for (int c = 0; c < cdh->latest_count; c++) {
			for (int ca_i = 0; ca_i < cc->camera_count; ca_i++) {
				if (ca_i == ca) continue;
				struct cam_detection_history* cdh_i = &cc->cam_awareness[ca_i].detection_history;
				for (int c_i = 0; c_i < cdh_i->latest_count; c_i++) {
					sdm3->class_match_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i] = (
							sdm3->detections_3d[ca * sdm3->cdh_max_size + c].class_id == sdm3->detections_3d[ca_i * sdm3->cdh_max_size + c_i].class_id
						);

					if (!sdm3->class_match_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i]) continue;

					struct vector3<float> pmp = cc->cam_awareness[ca_i].calibration.position - cc->cam_awareness[ca].calibration.position;

					struct vector3<float> u = sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction;
					struct vector3<float> v = sdm3->detections_3d[ca_i * sdm3->cdh_max_size + c_i].direction;

					struct vector3<float> vxu = cross(v, u);
					float len_vxu = length(vxu);

					if (len_vxu > 0) {
						sdm3->distance_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i] = scalar_proj(pmp, vxu);
						float t = -dot(cross(pmp, u), vxu) / length(vxu);
						float s = -dot(cross(pmp, v), vxu) / length(vxu);
						struct vector3<float> v_i = cc->cam_awareness[ca_i].calibration.position - v * -t;
						struct vector3<float> u_i = cc->cam_awareness[ca].calibration.position - u * -s;
						sdm3->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i] = v_i - ((u_i - v_i) * -0.5f);
						/*
						sdm3->size_factor_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i] = (
								sdm3->detections_3d[ca_i * sdm3->cdh_max_size + c_i].dimensions[1] / (t * distance_unit)
								/
								sdm3->detections_3d[ca * sdm3->cdh_max_size + c].dimensions[1] / (s * distance_unit)
							);
						*/
					} else {
						sdm3->class_match_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i] = false;
						//min dist of parallel lines
						//sdm3->matches[sm].distance[ca] = length(cross(v, pmp)) / length(v);
					}
				}
			}
		}
	}

	memset(sdm3->is_matched, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
	memset(sdm3->detections_buffer, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(struct cam_detection_3d));

	bool* used_current = (bool*)malloc(cc->camera_count * sdm3->cdh_max_size);

	//assign ray groups
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		for (int c = 0; c < cdh->latest_count; c++) {
			if (sdm3->is_matched[ca * sdm3->cdh_max_size + c]) continue;
			float inv_best_score = 0.0f;
			memset(used_current, 0, sizeof(cc->camera_count * sdm3->cdh_max_size));
			used_current[ca * sdm3->cdh_max_size + c] = true;
			int used_count = 0;
			struct statistic_detection_matcher_3d_detection* sdm3dd = &sdm3->detections_3d[ca * sdm3->cdh_max_size + c];
			for (int ca_i = 0; ca_i < cc->camera_count; ca_i++) {
				if (ca_i == ca) continue;
				struct cam_detection_history* cdh_i = &cc->cam_awareness[ca_i].detection_history;

				int best_score_idx = -1;
				float best_dist = 1000000.0f;

				for (int c_i = 0; c_i < cdh_i->latest_count; c_i++) {
					if (!sdm3->is_matched[ca_i * sdm3->cdh_max_size + ca_i]) {
						if (sdm3->class_match_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i]) {
							if (sdm3->distance_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i] < best_dist) {
								best_score_idx = c_i;
								best_dist = sdm3->distance_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i];
							}
						}
					}
				}
				if (best_score_idx > -1) {
					float some_threshold = 10000.0f;
					if (best_dist < some_threshold) {
						used_current[ca_i * sdm3->cdh_max_size + best_score_idx] = true;
						//TMP
						inv_best_score += best_dist;
						used_count++;
						sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].position = (sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].position - -sdm3->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + best_score_idx]);
					}
				}
			}
			if (used_count > 0) {
				sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].class_id = sdm3dd->class_id;
				sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].position = sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].position * (1.0f / (float)used_count);
				sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].timestamp = sdm3dd->timestamp;
				sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].score = inv_best_score;
				for (int ui = 0; ui < cc->camera_count * sdm3->cdh_max_size; ui++) {
					if (used_current[ui]) sdm3->is_matched[ui] = true;
				}
			}
		}
	}
	
	memset(sdm3->is_final_matched, 0, cc->camera_count* sdm3->cdh_max_size * sizeof(bool));

	//match ray groups with existing ray group or make new one
	for (int d = 0; d < sdm3->size; d++) {
		int best_idx = -1;
		float inv_best_score = 10000000.0f;
		int free_tracker_idx = -1;
		if (sdm3->detections[d].timestamp > 0) {
			for (int d_i = 0; d_i < cc->camera_count * sdm3->cdh_max_size; d_i++) {
				if (sdm3->is_final_matched[d_i]) continue;
				if (sdm3->detections_buffer[d_i].timestamp - sdm3->detections[d].timestamp > sdm3->ttl) {
					sdm3->detections[d].timestamp = 0;
					if (free_tracker_idx < 0) {
						free_tracker_idx = d;
						break;
					}
				} else {
					if (sdm3->detections_buffer[d_i].class_id == sdm3->detections[d].class_id) {
						struct vector3<float> pos_diff = sdm3->detections_buffer[d_i].position - sdm3->detections[d].position;

						//normalise
						float					t_diff = sdm3->detections_buffer[d_i].timestamp - sdm3->detections[d].timestamp;

						float det_dist = length(pos_diff);

						//make time dependent
						struct vector3<float> vel = pos_diff;

						float vel_dist = length(sdm3->detections[d].velocity - vel);

						//TODO: some velocity based threshold
						if (det_dist < inv_best_score) {
							best_idx = 0;
							inv_best_score = det_dist;
						}
					}
				}
			}
			if (best_idx > -1) {
				sdm3->is_final_matched[best_idx] = true;

				//make time dependent
				sdm3->detections[d].velocity = sdm3->detections_buffer[best_idx].position - sdm3->detections[d].position;
				sdm3->detections[d].position = sdm3->detections_buffer[best_idx].position;
				sdm3->detections[d].score = inv_best_score;
				sdm3->detections[d].timestamp = sdm3->detections_buffer[best_idx].timestamp;
			}
		} else {
			if (free_tracker_idx < 0) {
				free_tracker_idx = d;
			}
		}
	}

	for (int d_i = 0; d_i < cc->camera_count * sdm3->cdh_max_size; d_i++) {
		if (sdm3->is_final_matched[d_i]) continue;
		if (sdm3->detections_buffer[d_i].timestamp == 0) continue;
		for (int d = 0; d < sdm3->size; d++) {
			if (sdm3->detections[d].timestamp == 0) {
				sdm3->is_final_matched[d_i] = true;
				sdm3->detections[d].class_id = sdm3->detections_buffer[d_i].class_id;
				sdm3->detections[d].velocity = { 0.0f, 0.0f, 0.0f };
				sdm3->detections[d].position = sdm3->detections_buffer[d_i].position;
				sdm3->detections[d].score = 0.0f;
				sdm3->detections[d].timestamp = sdm3->detections_buffer[d_i].timestamp;
				break;
			}
		}
	}
}
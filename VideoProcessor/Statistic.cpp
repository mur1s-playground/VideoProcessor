#include "Statistic.h"

#include <math.h>
#include <cstdlib>
#include "Vector2.h"
#include "Logger.h"
#include "Util.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

#include "StatisticsKernel.h"

#include <algorithm>

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

void statistic_detection_matcher_3d_init(struct statistic_detection_matcher_3d* sdm3, int size, unsigned long long ttl, struct camera_control* cc, int population_c) {
	sdm3->detections = (struct cam_detection_3d*)malloc(size*sizeof(struct cam_detection_3d));
	memset(sdm3->detections, 0, size * sizeof(struct cam_detection_3d));

	sdm3->cdh_max_size = 1;
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		if (cdh->size > sdm3->cdh_max_size) {
			sdm3->cdh_max_size = cdh->size;
		}
	}

	sdm3->is_final_matched = (bool*)malloc(cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
	memset(sdm3->is_final_matched, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(bool));

	sdm3->size = size;
	sdm3->ttl = ttl;
	

	sdm3->is_matched = (bool*)malloc(cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
	sdm3->detections_buffer = (struct cam_detection_3d*)malloc(cc->camera_count * sdm3->cdh_max_size * sizeof(struct cam_detection_3d));
	sdm3->detections_3d = (struct statistic_detection_matcher_3d_detection*)malloc(cc->camera_count * sdm3->cdh_max_size * sizeof(struct statistic_detection_matcher_3d_detection));
	
								//class match matrix
	sdm3->memory_pool_size =	cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(bool) +
								//distance matrix
								cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float) + 
								//min dist central points matrix
								cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(struct vector3<float>) +
								//size facotr matrix
								cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float);
	sdm3->memory_pool = (unsigned char*)malloc(sdm3->memory_pool_size);
	cudaMalloc(&sdm3->memory_pool_device, sdm3->memory_pool_size);

	sdm3->class_match_matrix = (bool *)sdm3->memory_pool;//(bool*)malloc(cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
	sdm3->distance_matrix = (float*)(&sdm3->memory_pool[cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(bool)]); //(float *)malloc(cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float));
	sdm3->min_dist_central_points_matrix = (struct vector3<float> *)(&sdm3->memory_pool[cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(bool) + cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float)]); //(struct vector3<float> *) malloc(cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(struct vector3<float>));
	sdm3->size_factor_matrix = (float *)(&sdm3->memory_pool[cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(bool) + cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float) + cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(struct vector3<float>)]);//(float *)malloc(cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float));
	cudaMalloc(&sdm3->distance_matrix_device, cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float));

	sdm3->population_c		= population_c;							//object count	+ object genetic
	sdm3->population		= (int*) malloc(sdm3->population_c *	(1				+ (size			 * cc->camera_count)) * sizeof(int));
	sdm3->population_bak	= (int*) malloc(sdm3->population_c *	(1				+ (size			 * cc->camera_count)) * sizeof(int));
	cudaMalloc(&sdm3->population_device, sdm3->population_c * (1 + (size * cc->camera_count)) * sizeof(int));
	
	sdm3->population_scores = (float*)malloc(sdm3->population_c * sizeof(float));
	cudaMalloc(&sdm3->population_scores_device, sdm3->population_c * sizeof(float));

	sdm3->population_scores_idxs_orig = (int*)malloc(sdm3->population_c * sizeof(int));
	for (int i = 0; i < sdm3->population_c; i++) {
		sdm3->population_scores_idxs_orig[i] = i;
	}
	sdm3->population_scores_idxs = (int*)malloc(sdm3->population_c * sizeof(int));

	cudaMalloc(&sdm3->population_evolution_buffer_device, sdm3->population_c * (2 * sdm3->size + 2 * cc->camera_count * sdm3->cdh_max_size) * sizeof(unsigned char));

	sdm3->population_max_evolutions = 10;
	sdm3->population_keep_factor = 1.0f / 4.0f;
	sdm3->population_mutation_rate = 0.02f;

	sdm3->randoms_size = 1024;
	sdm3->randoms = (float * )malloc(sdm3->randoms_size * sizeof(float));
	cudaMalloc(&sdm3->randoms_device, sdm3->randoms_size * sizeof(float));
}

int tmp_ct = 0;

void statistic_evolutionary_tracker_population_init(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, int min_objects, int max_objects, int cdh_current) {
	int* rays_per_cam = (int*)malloc(cc->camera_count * sizeof(int));
	int* rays_per_cam_wrk = (int*)malloc(cc->camera_count * sizeof(int));

	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		rays_per_cam[ca] = cdh->latest_count;
	}

	int *init_member = (int *) malloc((1 + (sdm3->size * cc->camera_count)) * sizeof(int));
	init_member[0] = 0;
	for (int o = 1; o < 1 + (sdm3->size * cc->camera_count); o++) {
		init_member[o] = -1;
	}

	for (int p = 0; p < sdm3->population_c; p++) {
		memset(sdm3->is_matched, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
		memcpy(rays_per_cam_wrk, rays_per_cam, cc->camera_count * sizeof(int));

		int* object_count = &sdm3->population[p * (1 + (sdm3->size * cc->camera_count))];
		int* genetic_base_idx = object_count + 1;

		memcpy(object_count, init_member, (1 + (sdm3->size * cc->camera_count)) * sizeof(int));

		object_count[0] = min_objects + (int)((rand() / (float)RAND_MAX) * (float)(max_objects - min_objects));

		int rays_taken = 0;
		int tries = 0;
		float tries_factor = 1.0f;
		while (rays_taken < cdh_current) {
			tries_factor = 1.0f + (tries * 0.1f);
			for (int o = 0; o < object_count[0]; o++) {
				for (int r = 0; r < cc->camera_count; r++) {
					//if this cam has unassigned rays
					if (rays_per_cam_wrk[r]) {
						//slot free
						if (genetic_base_idx[o * cc->camera_count + r] == -1) {
							float p_r = (rand() / (float)RAND_MAX);
							//probability of ray being in that slot, speed up over time
							if (p_r <= (rays_per_cam[r] / (float)object_count[0]) * tries_factor) {
								//pick one
								for (int r_t = 0; r_t < rays_per_cam[r]; r_t++) {
									if (!sdm3->is_matched[r * sdm3->cdh_max_size + r_t]) {
										genetic_base_idx[o * cc->camera_count + r] = r_t;
										sdm3->is_matched[r * sdm3->cdh_max_size + r_t] = true;
										rays_per_cam_wrk[r]--;
										rays_taken++;
										break;
									}
								}
							}
						}
					}
				}
			}
			tries++;
		}
	}
	
	free(rays_per_cam);
	free(rays_per_cam_wrk);

	free(init_member);
}

//int iteration = 0;

void statistic_evolutionary_tracker_population_next_pool(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc) {
	float sum_scores = 0.0f;
	for (int p = 0; p < sdm3->population_c; p++) {
		if (sdm3->population_scores[p] >= 0.0f) {
			sum_scores += sdm3->population_scores[p];
		}
	}
	stable_sort(sdm3->population_scores_idxs, &sdm3->population_scores_idxs[sdm3->population_c], [sdm3](size_t i1, size_t i2) {return sdm3->population_scores[i1] < sdm3->population_scores[i2]; });
	logger(sdm3->population_scores[sdm3->population_scores_idxs[0]]);
	/*
	logger("iteration", iteration);
	for (int i = 0; i < sdm3->population_c; i++) {
		logger(sdm3->population_scores[sdm3->population_scores_idxs[i]]);
	}
	if (iteration == 1) {
		exit(0);
	}
	iteration++;
	*/

	int *pop_cur = sdm3->population;
	int *pop_next = sdm3->population_bak;
	if (sdm3->population_swap) {
		pop_next = pop_cur;
		pop_cur = sdm3->population_bak;
	}

	int population_kept_c = (int)floorf(sdm3->population_c * sdm3->population_keep_factor);
	int total_new = 0;
	for (int p = 0; p < population_kept_c; p++) {
		int target_idx = total_new * (1 + (sdm3->size * cc->camera_count));
		int origin_idx = sdm3->population_scores_idxs[p] * (1 + (sdm3->size * cc->camera_count));
		memcpy(&pop_next[target_idx], &pop_cur[origin_idx], (1 + (sdm3->size * cc->camera_count) * sizeof(int)));
		total_new++;
	}
	int tries = 0;
	float tries_factor = 1.0f;

	//min score = best score 0.0 err on all triangulations, max score = worst score...
	float min_score = sdm3->population_scores[sdm3->population_scores_idxs[0]];
	float max_kept_score = sdm3->population_scores[sdm3->population_scores_idxs[population_kept_c - 1]];
	if (max_kept_score == min_score) max_kept_score += 1e-9;

	while (total_new < sdm3->population_c) {
		tries_factor = 1.0f * ((tries/(sdm3->population_c - population_kept_c)) * 0.1f);
		int p_i = (int)(((rand() / (float)RAND_MAX)) * (sdm3->population_c * sdm3->population_keep_factor));
		float i_score = sdm3->population_scores[sdm3->population_scores_idxs[p_i]];
		float p_in = 1.0f - ((i_score - min_score) / (max_kept_score - min_score));
		float p_roll = (rand() / (float)RAND_MAX);
		if (p_roll <= p_in * tries_factor) {
			int target_idx = total_new * (1 + (sdm3->size * cc->camera_count));
			int origin_idx = sdm3->population_scores_idxs[p_i] * (1 + (sdm3->size * cc->camera_count));
			memcpy(&pop_next[target_idx], &pop_cur[origin_idx], (1 + (sdm3->size * cc->camera_count) * sizeof(int)));
			total_new++;
		}
		tries++;
	}
}

void statistic_detection_matcher_3d_update(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, struct camera_control_shared_state *ccss) {
	float distance_unit = cc->cam_awareness[0].calibration.d_1;

	int cdh_max_c = 0;
	int cdh_currents = 0;
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		cdh_currents += cdh->latest_count;
		if (cdh->latest_count > cdh_max_c) {
			cdh_max_c = cdh->latest_count;
		}
	}

	if (cdh_currents == 0) return;

	unsigned long long t_now = 0;

	//precompute ray meta
	int shared_rays = 0;
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		if (ccss != nullptr) {
			memset(&ccss[ca].latest_detections_rays, 0, 15 * sizeof(struct vector3<float>));
			memset(&ccss[ca].latest_detections_rays_origin, 0, 15 * sizeof(struct vector3<float>));
		}
		for (int c = 0; c < cdh->latest_count; c++) {
			int cur_h_idx = cdh->latest_idx - c;
			if (cur_h_idx < 0) cur_h_idx += cdh->size;
			struct cam_detection* current_detection = &cdh->history[cur_h_idx];

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].class_id = current_detection->class_id;

			struct vector2<float> det_center = cam_detection_get_center(current_detection);

			float north_pole = cc->cam_awareness[ca].north_pole.angle;
			float horizon = cc->cam_awareness[ca].horizon.angle;

			struct vector2<int> diff_from_mid = {
				(int)(cc->cam_awareness[ca].resolution_offset[0] + cc->cam_meta[ca].resolution[0] / 2.0f - det_center[0]),
				(int)(cc->cam_awareness[ca].resolution_offset[1] + cc->cam_meta[ca].resolution[1] / 2.0f - det_center[1])
			};

			float fov_np = cc->cam_awareness[ca].calibration.lens_fov[0];
			float fov_h = cc->cam_awareness[ca].calibration.lens_fov[1];

			struct vector2<float> angles_vec = {
				north_pole - (diff_from_mid[0] / (cc->cam_meta[ca].resolution[0] / 2.0f)) * (fov_np / 2.0f) - 90.0f,
				horizon - (diff_from_mid[1] / (cc->cam_meta[ca].resolution[1] / 2.0f)) * (fov_h / 2.0f) + 90.0f
			};
			if (angles_vec[0] < 0.0f) angles_vec[0] += 360.0f;
			if (angles_vec[1] < 0.0f) angles_vec[1] += 360.0f;
			if (angles_vec[0] > 360.0f) angles_vec[0] -= 360.0f;
			if (angles_vec[1] > 360.0f) angles_vec[1] -= 360.0f;

			float M_PI = 3.14159274101257324219;

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction = {
				sinf(angles_vec[1] * M_PI / (2.0f * 90.0f)) * cosf(angles_vec[0] * M_PI / (2.0f * 90.0f)),
				-sinf(angles_vec[1] * M_PI / (2.0f * 90.0f)) * sinf(angles_vec[0] * M_PI / (2.0f * 90.0f)),
				cosf(angles_vec[1] * M_PI / (2.0f * 90.0f))
			};
			/*
			if (tmp_ct == 100) {
				logger("detection_direction");
				logger("np", angles_vec[0] );
				logger("horizon", angles_vec[1] - 90.0f);
				logger("ca", ca);
				logger("ca", c);
				logger("d_x", sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction[0]);
				logger("d_y", sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction[1]);
				logger("d_z", sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction[2]);
				logger("--------------------");
			}
			*/

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].dimensions = {
				current_detection->x2 - current_detection->x1,
				current_detection->y2 - current_detection->y1
			};

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].timestamp = current_detection->timestamp;
			t_now = current_detection->timestamp;

			if (ccss != nullptr && c < 5) {
				ccss[shared_rays / 15].latest_detections_rays_origin[shared_rays % 15] = cc->cam_awareness[ca].calibration.position;
				ccss[shared_rays / 15].latest_detections_rays[shared_rays % 15] = sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction;
				shared_rays++;

				//ccss[ca].latest_detections_rays[c] = {angles_vec[0] + 180.0f, angles_vec[1] };
			}
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
					sdm3->class_match_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = (
							sdm3->detections_3d[ca * sdm3->cdh_max_size + c].class_id == sdm3->detections_3d[ca_i * sdm3->cdh_max_size + c_i].class_id
						);

					if (!sdm3->class_match_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i]) continue;

					struct vector3<float> pmp = cc->cam_awareness[ca_i].calibration.position - cc->cam_awareness[ca].calibration.position;

					struct vector3<float> u = sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction;
					struct vector3<float> v = sdm3->detections_3d[ca_i * sdm3->cdh_max_size + c_i].direction;

					struct vector3<float> vxu = cross(v, u);
					float len_vxu = length(vxu);

					if (len_vxu > 0) {
						sdm3->distance_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = abs(scalar_proj(pmp, vxu));
						float t = -dot(cross(pmp, u), vxu) / length(vxu);
						float s = -dot(cross(pmp, v), vxu) / length(vxu);
						/*
						logger("ca", ca);
						logger("c", c);
						logger("ca_i", ca_i);
						logger("c_i", c_i);
						logger("t", t);
						logger("s", s);
						*/
						struct vector3<float> v_i = cc->cam_awareness[ca_i].calibration.position - v * -t;
						struct vector3<float> u_i = cc->cam_awareness[ca].calibration.position - u * -s;
						sdm3->min_dist_central_points_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = v_i - ((u_i - v_i) * -0.5f);
						/*
						logger("cp_x", sdm3->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i][0]);
						logger("cp_y", sdm3->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i][1]);
						logger("cp_z", sdm3->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i][2]);
						*/
						/*
						sdm3->size_factor_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i] = (
								sdm3->detections_3d[ca_i * sdm3->cdh_max_size + c_i].dimensions[1] / (t * distance_unit)
								/
								sdm3->detections_3d[ca * sdm3->cdh_max_size + c].dimensions[1] / (s * distance_unit)
							);
						*/
					} else {
						sdm3->class_match_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = false;
						//min dist of parallel lines
						//sdm3->matches[sm].distance[ca] = length(cross(v, pmp)) / length(v);
					}
				}
			}
		}
	}
	//cudaMemcpyAsync(sdm3->memory_pool_device, sdm3->memory_pool, sdm3->memory_pool_size, cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaMemcpyAsync(sdm3->distance_matrix_device, sdm3->distance_matrix, cc->camera_count* sdm3->cdh_max_size* cc->camera_count* sdm3->cdh_max_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	
	int max_objects = min(cdh_currents, sdm3->size);
	int min_objects = max((int)ceilf(cdh_currents / (float)cc->camera_count), cdh_max_c);

	statistic_evolutionary_tracker_population_init(sdm3, cc, min_objects, max_objects, cdh_currents);

	int* pop_cur = sdm3->population;
	int* pop_next = sdm3->population_bak;

	sdm3->population_swap = false;
	for (int e = 0; e < sdm3->population_max_evolutions; e++) {
		pop_cur = sdm3->population;
		pop_next = sdm3->population_bak;
		if (sdm3->population_swap) {
			pop_next = pop_cur;
			pop_cur = sdm3->population_bak;
		}

		cudaMemcpyAsync(sdm3->population_device, pop_cur, sdm3->population_c * (1 + (sdm3->size * cc->camera_count)) * sizeof(int), cudaMemcpyHostToDevice, cuda_streams[0]);

		
		for (int ra = 0; ra < sdm3->randoms_size; ra++) {
			sdm3->randoms[ra] = (rand() / (float)RAND_MAX);
		}
		cudaMemcpyAsync(sdm3->randoms_device, sdm3->randoms, sdm3->randoms_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
		
		memcpy(sdm3->population_scores_idxs, sdm3->population_scores_idxs_orig, sdm3->population_c * sizeof(int));

		cudaStreamSynchronize(cuda_streams[0]);
		
		if (e > 0) {
			cudaMemsetAsync(sdm3->population_evolution_buffer_device, 0, sdm3->population_c* (2 * sdm3->size + 2 * cc->camera_count * sdm3->cdh_max_size) * sizeof(unsigned char), cuda_streams[3]);
			statistics_evolutionary_tracker_population_evolve_kernel_launch(sdm3->size, cc->camera_count, sdm3->cdh_max_size, sdm3->population_device, sdm3->population_evolution_buffer_device, sdm3->population_scores_device, sdm3->population_c, sdm3->population_keep_factor, sdm3->population_mutation_rate, sdm3->randoms_device, sdm3->randoms_size, min_objects, max_objects);
		}
		
		statistics_evolutionary_tracker_kernel_launch(sdm3->distance_matrix_device, sdm3->size, cc->camera_count, sdm3->cdh_max_size, sdm3->population_device, sdm3->population_scores_device, sdm3->population_c);
		
		cudaMemcpyAsync(sdm3->population_scores, sdm3->population_scores_device, sdm3->population_c * sizeof(float), cudaMemcpyDeviceToHost, cuda_streams[4]);
		cudaStreamSynchronize(cuda_streams[4]);
		
		if (e < sdm3->population_max_evolutions - 1) {
			statistic_evolutionary_tracker_population_next_pool(sdm3, cc);
			sdm3->population_swap = !sdm3->population_swap;
		}

	}
	
	stable_sort(sdm3->population_scores_idxs, &sdm3->population_scores_idxs[sdm3->population_c], [sdm3](size_t i1, size_t i2) {return sdm3->population_scores[i1] < sdm3->population_scores[i2]; });
	int object_idx = sdm3->population_scores_idxs[0] * (1 + (sdm3->size * cc->camera_count));
	int* object_it = &pop_cur[object_idx];
	int object_count = object_it[0];
	object_it++;
	int shared_objects = 0;
	for (int o = 0; o < object_count; o++) {
		sdm3->detections[o].class_id = 37;
		sdm3->detections[o].velocity = { 0.0f, 0.0f, 0.0f };
		sdm3->detections[o].position = { 0.0f, 0.0f, 0.0f };
		int ray_count = 0;
		int c_id_last = -1;
		int r_id_last = -1;
		for (int r = 0; r < cc->camera_count; r++) {
			int c_id = r;
			int r_id = object_it[0];
			object_it++;
			if (r_id > -1) {
				if (c_id_last > -1) {
					sdm3->detections[o].position = sdm3->detections[o].position - -sdm3->min_dist_central_points_matrix[c_id_last * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + r_id_last * cc->camera_count * sdm3->cdh_max_size +  c_id * sdm3->cdh_max_size + r_id];
					ray_count++;
				}
				c_id_last = c_id;
				r_id_last = r_id;
			}
		}
		if (ray_count > 0) {
			sdm3->detections[o].position = sdm3->detections[o].position / (float)ray_count;
			sdm3->detections[o].score = 0.0f;
			sdm3->detections[o].timestamp = t_now;
			/*
			logger("o", o);
			logger("position_x", sdm3->detections[o].position[0]);
			logger("position_y", sdm3->detections[o].position[1]);
			logger("position_z", sdm3->detections[o].position[2]);
			*/
			if (ccss != nullptr && shared_objects < cc->camera_count * 5) {
				ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = sdm3->detections[o].position;
				shared_objects++;
			}
		}
	}

	/*
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
			sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].ray_position[used_count] = cc->cam_awareness[ca].calibration.position;
			sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].ray_direction[used_count] = sdm3dd->direction;
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
						if (used_count < 5) {
							sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].ray_position[used_count] = cc->cam_awareness[ca_i].calibration.position;
							sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].ray_direction[used_count] = sdm3->detections_3d[ca_i * sdm3->cdh_max_size + best_score_idx].direction;
						}
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

	int shared_objects = 0;
	if (ccss != nullptr) {
		for (int ca = 0; ca < cc->camera_count; ca++) {
			memset(&ccss[ca].latest_detections_objects, 0, 5 * sizeof(struct vector3<float>));
		}
	}

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
				if (ccss != nullptr && shared_objects < cc->camera_count * 5) {
					ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = sdm3->detections[d].position;
					shared_objects++;
				}
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

				if (ccss != nullptr && shared_objects < cc->camera_count * 5) {
					ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = sdm3->detections[d].position;
					shared_objects++;
				}

				break;
			}
		}
	}
	*/
}

void statistic_quantized_grid_init(struct statistic_quantized_grid* sqg, std::vector<struct vector2<int>> spans, std::vector<float> quantization_factors, int data_size, void **data) {
	sqg->dim_c = spans.size();
	sqg->spans = (struct vector2<int>*)malloc(sqg->dim_c * sizeof(struct vector2<int>));
	memcpy(sqg->spans, spans.data(), sqg->dim_c * sizeof(struct vector2<int>));
	sqg->quantization_factors = (float*)malloc(sqg->dim_c * sizeof(float));
	memcpy(sqg->quantization_factors, quantization_factors.data(), sqg->dim_c * sizeof(float));
	sqg->data_size = data_size;
	sqg->dimensions = (int*)malloc(sqg->dim_c * sizeof(int));
	sqg->total_size = 1;
	for (int d = 0; d < sqg->dim_c; d++) {
		sqg->dimensions[d] = (int)ceilf((sqg->spans[d][1] - sqg->spans[d][0]) * quantization_factors[d]);
		sqg->total_size *= sqg->dimensions[d];
	}
	sqg->total_size *= sqg->data_size;
	sqg->data = (void*)malloc(sqg->total_size);

	memset(sqg->data, 0, sqg->total_size);

	*data = sqg->data;
}

int statistic_quantized_grid_get_base_idx(struct statistic_quantized_grid* sqg, float* position) {
	int idx = 0;
	int factor_offset = 1;
	for (int d = sqg->dim_c - 1; d >= 0; d--) {
		if (d < sqg->dim_c - 1)	factor_offset *= sqg->dimensions[d+1];

		int d_idx = (int)floorf((position[d] - sqg->spans[d][0]) * sqg->quantization_factors[d]);
		if (d_idx < 0 || d_idx >= sqg->dimensions[d]) return -1;

		idx += (factor_offset * d_idx);
	}
	return idx;
}

int statistic_quantized_grid_get_directional_base_idx(struct statistic_quantized_grid* sqg, float* position, float* direction) {
	float lambda_min = 10000000.0f;
	for (int d = 0; d < sqg->dim_c; d++) {
		if (direction[d] != 0) {
			float lambda = (floorf((position[d] - sqg->spans[d][0]) * sqg->quantization_factors[d]) + (1 - 2 * (direction[d] < 0)) - position[d] - sqg->spans[d][0]) / direction[d];
			if (lambda < lambda_min) lambda_min = lambda;
		}
	}
	int idx = -1;
	if (lambda_min == 10000000.0f) {
		int idx = statistic_quantized_grid_get_base_idx(sqg, position);
	} else {
		float* pos = (float*)malloc(sqg->dim_c * sizeof(float));
		for (int d = 0; d < sqg->dim_c; d++) {
			pos[d] = position[d] + lambda_min * direction[d];
		}
		int idx = statistic_quantized_grid_get_base_idx(sqg, pos);
		free(pos);
	}
	return idx;
}

void statistic_heatmap_init(struct statistic_heatmap* sh, struct vector2<int> x_dim, struct vector2<int> y_dim, struct vector2<int> z_dim, struct vector3<float> quantization_factors, float falloff, string save_load_dir) {
	statistic_quantized_grid_init(&sh->sqg, std::vector<struct vector2<int>>{ x_dim, y_dim, z_dim }, std::vector<float>{quantization_factors[0], quantization_factors[1], quantization_factors[2] }, sizeof(float), (void **)&sh->data);

	sh->save_load_dir = save_load_dir;

	stringstream ss_dir_file;
	ss_dir_file << sh->save_load_dir << "heatmap.bin";

	size_t read_len = 0;
	util_read_binary(ss_dir_file.str(), (unsigned char*)sh->data, &read_len);

	sh->known_max = 0.0f;

	if (read_len > 0) {
		for (int i = 0; i < sh->sqg.total_size / sh->sqg.data_size; i++) {
			if (sh->data[i] > sh->known_max) {
				sh->known_max = sh->data[i];
			}
		}
	}

	sh->falloff = falloff;

	cudaMalloc((void**)&sh->device_data, sh->sqg.total_size);
}

void statistic_heatmap_update(struct statistic_heatmap* sh, struct vector3<float> position) {
	int idx = statistic_quantized_grid_get_base_idx(&sh->sqg, (float *) &position);
	if (idx > -1) {
		sh->data[idx] += 1.0f;
		if (sh->data[idx] > sh->known_max) {
			sh->known_max = sh->data[idx];
		}
		sh->data[idx] /= sh->known_max;
	}
}

void statistic_heatmap_update_calculate(struct statistic_heatmap* sh) {
	statistics_heatmap_kernel_launch(sh->data, sh->device_data, struct vector3<int>(sh->sqg.dimensions[0], sh->sqg.dimensions[1], sh->sqg.dimensions[2]), sh->falloff);
}

void statistic_heatmap_save(struct statistic_heatmap* sh) {
	stringstream ss_dir_file;
	ss_dir_file << sh->save_load_dir << "heatmap.bin";

	util_write_binary(ss_dir_file.str(), (unsigned char *) sh->data, sh->sqg.total_size);
}

void statistic_vectorfield_3d_init(struct statistic_vectorfield_3d* sv3d, struct vector2<int> x_dim, struct vector2<int> y_dim, struct vector2<int> z_dim, struct vector3<float> quantization_factors, int parts, std::string save_load_dir) {
	statistic_quantized_grid_init(&sv3d->sqg, std::vector<struct vector2<int>>{ x_dim, y_dim, z_dim, struct vector2<int>(0, 27), struct vector2<int>(0, 3) }, std::vector<float>{quantization_factors[0], quantization_factors[1], quantization_factors[2], 1.0f, 1.0f }, sizeof(float), (void**)&sv3d->data);

	sv3d->save_load_dir = save_load_dir;
	sv3d->part_factor = (1.0f / parts);

	for (int x = 0; x < sv3d->sqg.dimensions[0]; x++) {
		for (int y = 0; y < sv3d->sqg.dimensions[1]; y++) {
			for (int z = 0; z < sv3d->sqg.dimensions[2]; z++) {
				float pos[5] = { (float)x, (float)y, (float)z, 0, 0 };

				int idx = statistic_quantized_grid_get_base_idx(&sv3d->sqg, (float*)&pos);

				for (int d = 0; d < 27; d++) {
					sv3d->data[idx]		= 1.0f / 27.0f;
					sv3d->data[idx + 1] = 0.0f;
					sv3d->data[idx + 2] = 0.0f;

					idx += 3;
				}
			}
		}
	}

	sv3d->max_vel = 0.000001f;
	sv3d->max_acc = 0.000001f;
	
	stringstream ss_dir_file;
	ss_dir_file << sv3d->save_load_dir << "vectorfield_3d.bin";

	size_t out_len = 0;
	util_read_binary(ss_dir_file.str(), (unsigned char *)sv3d->data, &out_len);
	if (out_len > 0) {
		int i = 0;
		while (i < sv3d->sqg.total_size / sv3d->sqg.data_size) {
			//sv3d->data[i]
			if (sv3d->data[i + 1] > sv3d->max_vel) {
				sv3d->max_vel = sv3d->data[i + 1];
			}
			if (sv3d->data[i + 2] > sv3d->max_acc) {
				sv3d->max_acc = sv3d->data[i + 2];
			}

			i += 3;
		}
	}

	cudaMalloc((void**)&sv3d->device_data, sv3d->sqg.total_size);
}

void statistic_vectorfield_3d_update(struct statistic_vectorfield_3d* sv3d, struct vector3<float> position, struct vector3<float> velocity, float velocity_t, float acceleration_t) {
	float pos[5] = { position[0], position[1], position[2], 0, 0 };

	int idx = statistic_quantized_grid_get_base_idx(&sv3d->sqg, (float *)&pos);
	if (idx > -1) {
		struct vector3<float> vel_c = velocity / length(velocity);

		float angle_min = 180.0f;
		int idx_inner = (9 + 3 + 1) * 3;
		if (length(velocity) != 0) {
			struct vector3<int> id = { -1, -1, -1 };
			struct vector3<float> dir_m = { 0.0f, 0.0f, 0.0f };
			for (int z = -1; z < 1; z++) {
				for (int y = -1; y < 1; y++) {
					for (int x = -1; x < 1; x++) {
						if (x != 0 || y != 0 || z != 0) {
							struct vector3<float> direction = { (float)x, (float)y, (float)z };
							direction = direction / length(direction);
							struct vector3<float> dxv = cross(direction, vel_c);
							float angle = length(dxv);
							if (angle_min > angle) {
								angle_min = angle;
								id = { x, y, z };
								dir_m = direction;
								idx_inner = ((z + 1) * 9 + (y + 1) * 3 + (x + 1)) * 3;
							}
						}
					}
				}
			}
			if (length(dir_m - -vel_c) < length((dir_m * -1.0f) - -vel_c)) {
				idx_inner = ((-id[2] + 1) * 9 + (-id[1] + 1) * 3 + -id[0] + 1) * 3;
			}
		}
		float parts = 0.0f;
		for (int i = 0; i < 27; i++) {
			if (i != idx_inner) {
				float val = sv3d->part_factor*(1.0f / 26.0f) * sv3d->data[idx + (i * 3)];
				sv3d->data[idx + (i * 3)] -= val;
				parts += val;
			}
		}

		sv3d->data[idx + idx_inner] += parts;

		//TMP
		sv3d->data[idx + idx_inner + 1] = ((99.0f / 100.0f) * sv3d->data[idx + idx_inner + 1] + (1.0f / 100.0f) * velocity_t);
		if (sv3d->data[idx + idx_inner + 1] > sv3d->max_vel) sv3d->max_vel = sv3d->data[idx + idx_inner + 1];

		sv3d->data[idx + idx_inner + 2] = ((99.0f / 100.0f) * sv3d->data[idx + idx_inner + 2] + (1.0f / 100.0f) * acceleration_t);
		if (sv3d->data[idx + idx_inner + 2] > sv3d->max_acc) sv3d->max_acc = sv3d->data[idx + idx_inner + 2];
	}
}

void statistic_vectorfield_3d_update_device(struct statistic_vectorfield_3d* sv3d) {
	cudaMemcpyAsync(sv3d->device_data, sv3d->data, sv3d->sqg.total_size, cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaStreamSynchronize(cuda_streams[0]);
}

void statistic_vectorfield_3d_save(struct statistic_vectorfield_3d* sv3d) {
	stringstream ss_dir_file;
	ss_dir_file << sv3d->save_load_dir << "vectorfield_3d.bin";

	util_write_binary(ss_dir_file.str(), (unsigned char*) sv3d->data, sv3d->sqg.total_size);
}
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

bool statistic_angle_denoise_is_left_of(float angle_base, float angle_new) {
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
		float dt_s = 1 - (2 * (statistic_angle_denoise_is_left_of(sad->angle_distribution[sad->angle_distribution_idx_latest], sad->angle_distribution[tmp_d_c])));		
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
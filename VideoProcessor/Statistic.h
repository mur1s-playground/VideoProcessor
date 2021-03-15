#pragma once

#include "CameraControl.h"

struct statistic_angle_denoiser {
	float				angle;
	float				angle_stability;

	float*				angle_distribution;
	int					angle_distribution_size;
	int					angle_distribution_idx_latest;

	float*				angle_distribution_weights;
};

void statistic_angle_denoiser_init(struct statistic_angle_denoiser* sad, int size);
void statistic_angle_denoiser_set_weights(struct statistic_angle_denoiser* sad, float* weights);
void statistic_angle_denoiser_update(struct statistic_angle_denoiser* sad, float angle);

struct statistic_detection_matcher_2d {
	struct cam_detection* detections;
	struct cam_detection_history* matches_history;

	int size;
	unsigned long long ttl;
};

void statistic_detection_matcher_2d_init(struct statistic_detection_matcher_2d* sdm2, int size, unsigned long long ttl, int avg_size);
void statistic_detection_matcher_2d_update(struct statistic_detection_matcher_2d* sdm2, struct cam_detection_history* cdh);
int statistic_detection_matcher_2d_get_stable_match(struct statistic_detection_matcher_2d* sdm2, int class_id, int count_threshold);
#pragma once

#include "CameraControl.h"
/*
struct statistic_angle_denoiser {
	float				angle;
	float				angle_stability;

	float*				angle_distribution;
	int					angle_distribution_size;
	int					angle_distribution_idx_latest;

	float*				angle_distribution_weights;
};
*/

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

struct statistic_detection_matcher_matrix {
	float*					distance;
	struct vector3<float>	min_dist_central_points;
};

struct statistic_detection_matcher_3d_detection {
	int						class_id;

	struct vector3<float>	direction;

	struct vector2<int>		dimensions;

	unsigned long long		timestamp;
};

struct statistic_detection_matcher_3d {
	struct cam_detection_3d* detections;

	int size;
	unsigned long long ttl;

	bool *is_matched;
	struct cam_detection_3d* detections_buffer;

	bool* is_final_matched;
	struct statistic_detection_matcher_3d_detection* detections_3d;

	int cdh_max_size;

	bool* class_match_matrix;
	float* distance_matrix;
	struct vector3<float>* min_dist_central_points_matrix;
	float* size_factor_matrix;
	/*
	int match_max_cameras_used;
	struct statistic_detection_matcher_3d_match* matches;
	*/
};

void statistic_detection_matcher_3d_init(struct statistic_detection_matcher_3d* sdm3, int size, unsigned long long ttl, struct camera_control* cc);
void statistic_detection_matcher_3d_update(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, struct camera_control_shared_state* ccss);
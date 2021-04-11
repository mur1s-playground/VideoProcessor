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
bool statistic_angle_denoiser_is_left_of(float angle_base, float angle_new);

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

struct statistic_camera_ray_data {
	int						cdh_max_size;

	struct statistic_detection_matcher_3d_detection* detections_3d;

	bool*					class_match_matrix;
	bool*					class_match_matrix_device;

	float*					distance_matrix;
	float*					distance_matrix_device;

	struct vector3<float>*	ray_matrix;
	struct vector3<float>*	ray_matrix_device;

	int						single_ray_max_estimates;
	struct vector2<float>*	single_ray_position_estimate_device;

	struct vector3<float>*	min_dist_central_points_matrix;
	struct vector3<float>*	min_dist_central_points_matrix_device;

	float*					size_estimation_correction_dist_matrix;
	float*					size_estimation_correction_dist_matrix_device;
};

void statistic_camera_ray_data_init(struct statistic_camera_ray_data* scrd, struct camera_control *cc);

struct statistic_detection_matcher_3d {
	struct cam_detection_3d* detections;

	int size;
	unsigned long long ttl;

	bool *is_matched;
	struct cam_detection_3d* detections_buffer;

	bool* is_final_matched;
	//struct statistic_detection_matcher_3d_detection* detections_3d;

	int cdh_max_size;

	struct statistic_camera_ray_data	*scrd;
	/*
	bool* class_match_matrix;

	float* distance_matrix;
	float* distance_matrix_device;

	struct vector3<float>*  ray_matrix;
	struct vector3<float>*  ray_matrix_device;

	int						single_ray_max_estimates;
	struct vector2<float>*  single_ray_position_estimate_device;

	struct vector3<float>* min_dist_central_points_matrix;
	struct vector3<float>* min_dist_central_points_matrix_device;

	float* size_estimation_correction_dist_matrix;
	float* size_estimation_correction_dist_matrix_device;
	*/

	int					population_c;
	float*				population_scores;
	float*				population_scores_device;
	int*				population_scores_idxs_orig;
	int*				population_scores_idxs;

	float				population_keep_factor;
	int					population_max_evolutions;

	float				population_mutation_rate;

	unsigned char		*population_evolution_buffer_device;

	int					randoms_size;
	float*				randoms;
	float*				randoms_device;

	bool											population_swap;
	int*											population;
	int*											population_bak;

	int*											population_device;

	/*
	int match_max_cameras_used;
	struct statistic_detection_matcher_3d_match* matches;
	*/
};

void statistic_detection_matcher_3d_init(struct statistic_detection_matcher_3d* sdm3, int size, unsigned long long ttl, struct camera_control* cc, int population_c, struct statistic_camera_ray_data* scrd);
void statistic_detection_matcher_3d_update(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, struct camera_control_shared_state* ccss);

struct statistic_quantized_grid {
	int					dim_c;

	struct vector2<int>	*spans;
	float				*quantization_factors;
	int					*dimensions;

	int					data_size;
	int					total_size;

	void				*data;
};

void statistic_quantized_grid_init(struct statistic_quantized_grid *sqg, std::vector<struct vector2<int>> spans, std::vector<float> quantization_factors, int data_size, void **data);
int statistic_quantized_grid_get_base_idx(struct statistic_quantized_grid* sqg, float* position);

struct statistic_heatmap {
	struct statistic_quantized_grid		sqg;

	float				falloff;

	float*				data;
	float*				device_data;

	float				known_max;

	std::string			save_load_dir;
};

void statistic_heatmap_init(struct statistic_heatmap *sh, struct vector2<int> x_dim, struct vector2<int> y_dim, struct vector2<int> z_dim, struct vector3<float> quantization_fac, float falloff, std::string save_load_dir);
void statistic_heatmap_update(struct statistic_heatmap* sh, struct vector3<float> position);
void statistic_heatmap_update_calculate(struct statistic_heatmap* sh);
void statistic_heatmap_save(struct statistic_heatmap* sh);

struct statistic_vectorfield_3d {
	struct statistic_quantized_grid		sqg;

	float part_factor;

	float max_vel;
	float max_acc;

	float* data;
	float* device_data;

	std::string			save_load_dir;
};

void statistic_vectorfield_3d_init(struct statistic_vectorfield_3d *sv3d, struct vector2<int> x_dim, struct vector2<int> y_dim, struct vector2<int> z_dim, struct vector3<float> quantization_fac, int parts, std::string save_load_dir);
void statistic_vectorfield_3d_update(struct statistic_vectorfield_3d* sv3d, struct vector3<float> position, struct vector3<float> velocity, float velocity_t, float acceleration_t);
void statistic_vectorfield_3d_update_device(struct statistic_vectorfield_3d* sv3d);
void statistic_vectorfield_3d_save(struct statistic_vectorfield_3d* sv3d);

struct statistic_unscatter_interpolation_2d {
	struct vector2<int>		grid_size;
	struct vector2<int>		dimension;

	float					*data;
};

void statistic_unscatter_interpolation_init(struct statistic_unscatter_interpolation_2d *sui2d, struct vector2<int> grid_size, struct vector2<int> dimension);
void statistic_unscatter_interpolation_calculate(struct statistic_unscatter_interpolation_2d* sui2d, std::vector<struct vector2<float>> points, std::vector<float> values, int power);
void statistic_unscatter_interpolation_destroy(struct statistic_unscatter_interpolation_2d* sui2d);

struct statistic_unscatter_triangulation_2d {
	struct vector2<int>		grid_size;
	struct vector2<int>		dimension;

	float* data;
};

void statistic_unscatter_triangulation_init(struct statistic_unscatter_triangulation_2d* sut2d, struct vector2<int> grid_size, struct vector2<int> dimension);
void statistic_unscatter_triangulation_calculate(struct statistic_unscatter_triangulation_2d* sut2d, std::vector<struct vector2<float>> points, std::vector<float> values);
void statistic_unscatter_triangulation_destroy(struct statistic_unscatter_triangulation_2d* sut2d);

bool statistic_unscatter_triangulation_get_value(struct statistic_unscatter_triangulation_2d* sut2d, struct vector2<float> point, float* out_value);
float statistic_unscatter_triangulation_center_shift_inverse(struct statistic_unscatter_triangulation_2d* sut2d);
bool statistic_unscatter_triangulation_approximate_missing(struct statistic_unscatter_triangulation_2d* sut2d);

struct statistic_position_regression {
	struct vector3<float>	parameter_search_space;

	int						camera_c;
	struct vector3<float>	*camera_positions;

	string					temporary_storage_dir;

	int						t_samples_count;

	struct statistic_camera_ray_data	*scrd;
	int						t_current;
	int						t_c;
};

void statistic_position_regression_init(struct statistic_position_regression *spr, struct vector3<float> parameter_search_space, struct camera_control *cc, string temporary_storage_dir, struct statistic_camera_ray_data *scrd, int t_samples_count);
void statistic_position_regression_update(struct statistic_position_regression* spr, struct camera_control* cc);
void statistic_position_regression_calculate(struct statistic_postition_regression* spr);
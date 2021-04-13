#include "Statistic.h"

#include <math.h>
#include <cstdlib>
#include "Vector2.h"
#include "Logger.h"
#include "Util.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

#include "StatisticsKernel.h"

#include "Statistics3D.h"

#include <algorithm>
#include <limits>

float M_PI = 3.14159274101257324219;

// triangulation video dump
/*
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
*/

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

void statistic_camera_ray_data_init(struct statistic_camera_ray_data* scrm, struct camera_control* cc) {
	scrm->cdh_max_size = 1;

	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		if (cdh->size > scrm->cdh_max_size) {
			scrm->cdh_max_size = cdh->size;
		}
	}

	scrm->detections_3d = (struct statistic_detection_matcher_3d_detection*)malloc(cc->camera_count * scrm->cdh_max_size * sizeof(struct statistic_detection_matcher_3d_detection));
	cudaMalloc(&scrm->detections_3d_device, cc->camera_count * scrm->cdh_max_size * sizeof(struct statistic_detection_matcher_3d_detection));

	size_t matrix_base_size = cc->camera_count * scrm->cdh_max_size * cc->camera_count * scrm->cdh_max_size;

	scrm->class_match_matrix = (bool*)malloc(matrix_base_size * sizeof(bool));
	cudaMalloc(&scrm->class_match_matrix_device, matrix_base_size * sizeof(bool));

	scrm->distance_matrix = (float*)malloc(matrix_base_size * sizeof(float));
	cudaMalloc(&scrm->distance_matrix_device, matrix_base_size * sizeof(float));

	scrm->min_dist_central_points_matrix = (struct vector3<float> *) malloc(matrix_base_size * sizeof(struct vector3<float>));
	cudaMalloc(&scrm->min_dist_central_points_matrix_device, matrix_base_size * sizeof(struct vector3<float>));

	scrm->size_estimation_correction_dist_matrix = (float*)malloc(matrix_base_size * sizeof(float));
	cudaMalloc(&scrm->size_estimation_correction_dist_matrix_device, matrix_base_size * sizeof(float));

	scrm->ray_matrix = (struct vector3<float> *) malloc(cc->camera_count * (1 + scrm->cdh_max_size) * sizeof(struct vector3<float>));
	cudaMalloc(&scrm->ray_matrix_device, cc->camera_count * (1 + scrm->cdh_max_size) * sizeof(struct vector3<float>));

	scrm->single_ray_max_estimates = 5;
	cudaMalloc(&scrm->single_ray_position_estimate_device, scrm->single_ray_max_estimates * cc->camera_count * scrm->cdh_max_size * sizeof(struct vector2<float>));
}

unsigned long long statistic_camera_ray_data_convert_to_3d_detection(struct statistic_camera_ray_data* sdm3, struct camera_control* cc, struct camera_control_shared_state* ccss) {
	unsigned long long t_now = 0;

	memset(sdm3->detections_3d, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(struct statistic_detection_matcher_3d_detection));
	memset(sdm3->ray_matrix, 0, cc->camera_count * (1 + sdm3->cdh_max_size) * sizeof(struct vector3<float>));

	int shared_rays = 0;
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		sdm3->ray_matrix[ca * (1 + sdm3->cdh_max_size)] = cc->cam_awareness[ca].calibration.position;
		if (ccss != nullptr) {
			memset(&ccss[ca].latest_detections_rays, 0, 15 * sizeof(struct vector3<float>));
			memset(&ccss[ca].latest_detections_rays_origin, 0, 15 * sizeof(struct vector3<float>));
		}
		int c = 0;
		for (; c < cdh->latest_count; c++) {
			int cur_h_idx = cdh->latest_idx - c;
			if (cur_h_idx < 0) cur_h_idx += cdh->size;
			struct cam_detection* current_detection = &cdh->history[cur_h_idx];

			sdm3->detections_3d[ca * sdm3->cdh_max_size + c].class_id = current_detection->class_id;

			struct vector2<float> det_center = cam_detection_get_center(current_detection);

			float north_pole_offset = 0.0f;
			if (statistic_unscatter_triangulation_get_value(cc->cam_awareness[ca].calibration.lens_north_pole, struct vector2<float>(det_center[0] - cc->cam_awareness[ca].resolution_offset[0], det_center[1] - cc->cam_awareness[ca].resolution_offset[1]), &north_pole_offset)) {
				float horizon_offset = 0.0f;
				if (statistic_unscatter_triangulation_get_value(cc->cam_awareness[ca].calibration.lens_horizon, struct vector2<float>(det_center[0] - cc->cam_awareness[ca].resolution_offset[0], det_center[1] - cc->cam_awareness[ca].resolution_offset[1]), &horizon_offset)) {
					struct vector2<float> angles_vec = {
						cc->cam_awareness[ca].north_pole.angle + north_pole_offset - 90.0f,
						cc->cam_awareness[ca].horizon.angle + horizon_offset + 90.0f
					};

					if (angles_vec[0] < 0.0f) angles_vec[0] += 360.0f;
					if (angles_vec[1] < 0.0f) angles_vec[1] += 360.0f;
					if (angles_vec[0] > 360.0f) angles_vec[0] -= 360.0f;
					if (angles_vec[1] > 360.0f) angles_vec[1] -= 360.0f;

					sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction = {
						sinf(angles_vec[1] * M_PI / (2.0f * 90.0f)) * cosf(angles_vec[0] * M_PI / (2.0f * 90.0f)),
						-sinf(angles_vec[1] * M_PI / (2.0f * 90.0f)) * sinf(angles_vec[0] * M_PI / (2.0f * 90.0f)),
						cosf(angles_vec[1] * M_PI / (2.0f * 90.0f))
					};

					sdm3->ray_matrix[ca * (1 + sdm3->cdh_max_size) + c] = sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction;

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
					}

				}
			}
		}
	}
	return t_now;
}

int statistic_camera_ray_data_prepare_position_estimation_from_heatmap_async(struct statistic_camera_ray_data* sdm3, struct camera_control* cc, int cuda_stream_id) {
	cudaMemcpyAsync(sdm3->ray_matrix_device, sdm3->ray_matrix, cc->camera_count * (1 + sdm3->cdh_max_size) * sizeof(struct vector3<float>), cudaMemcpyHostToDevice, cuda_streams[cuda_stream_id]);

	gpu_memory_buffer_try_r(cc->statistics_3d_in->gmb, cc->statistics_3d_in->gmb->slots, true, 8);
	int heatmap_gpu_id = cc->statistics_3d_in->gmb->p_rw[2 * (cc->statistics_3d_in->gmb->slots + 1)];
	gpu_memory_buffer_release_r(cc->statistics_3d_in->gmb, cc->statistics_3d_in->gmb->slots);

	gpu_memory_buffer_try_r(cc->statistics_3d_in->gmb, heatmap_gpu_id, true, 8);
	vector3<int> heatmap_dimensions = { cc->statistics_3d_in->heatmap_3d.sqg.dimensions[0], cc->statistics_3d_in->heatmap_3d.sqg.dimensions[1], cc->statistics_3d_in->heatmap_3d.sqg.dimensions[2] };
	vector3<float> heatmap_quantization_factors = { cc->statistics_3d_in->heatmap_3d.sqg.quantization_factors[0], cc->statistics_3d_in->heatmap_3d.sqg.quantization_factors[1], cc->statistics_3d_in->heatmap_3d.sqg.quantization_factors[2] };
	vector3<int> heatmap_span_start = { cc->statistics_3d_in->heatmap_3d.sqg.spans[0][0], cc->statistics_3d_in->heatmap_3d.sqg.spans[1][0], cc->statistics_3d_in->heatmap_3d.sqg.spans[2][0] };

	float* heatmap_device_ptr = (float*)(cc->statistics_3d_in->gmb->p_device + (heatmap_gpu_id * cc->statistics_3d_in->gmb->size));
	statistics_evulotionary_tracker_single_ray_estimates_kernel_launch_async(sdm3->ray_matrix_device, cc->camera_count, sdm3->cdh_max_size, sdm3->single_ray_position_estimate_device, sdm3->single_ray_max_estimates, heatmap_dimensions, heatmap_quantization_factors, heatmap_span_start, heatmap_device_ptr, cuda_stream_id);

	return heatmap_gpu_id;
}

void statistic_camera_ray_data_wait_for_position_estimation_from_heatmap(struct camera_control* cc, int heatmap_gpu_id, int cuda_stream_id) {
	cudaStreamSynchronize(cuda_streams[cuda_stream_id]);
	gpu_memory_buffer_release_r(cc->statistics_3d_in->gmb, heatmap_gpu_id);
}

void statistic_camera_ray_data_precompute_ray_correlation(struct statistic_camera_ray_data* sdm3, struct camera_control* cc) {
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
						if (t <= 0 || s <= 0) {
							sdm3->class_match_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = false;
						}
						else {
							float width_t = t * 2.0f * tan(cc->cam_awareness[ca_i].calibration.lens_fov[0] * 0.5f * (M_PI/(180.0f))) * sdm3->detections_3d[ca_i * sdm3->cdh_max_size + c_i].dimensions[0] / (float)cc->cam_meta[ca_i].resolution[0];
							float width_s = s * 2.0f * tan(cc->cam_awareness[ca].calibration.lens_fov[0] * 0.5f * (M_PI / (180.0f))) * sdm3->detections_3d[ca * sdm3->cdh_max_size + c].dimensions[0] / (float)cc->cam_meta[ca].resolution[0];

							float target_t = width_s / (width_t / t);
							float target_s = width_t / (width_s / s);

							sdm3->size_estimation_correction_dist_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = (target_s - s);

							/*
							t = t + 0.5f * (target_t - t);
							s = s + 0.5f * (target_s - s);
							*/

							/*
							logger("w_t", width_t);
							logger("s_t", width_s);
							logger("t", t);
							logger("t_t", target_t);
							logger("s", s);
							logger("t_s", target_s);
							*/
							/*
							struct vector3<float> v_i = cc->cam_awareness[ca_i].calibration.position - v * -t;
							struct vector3<float> u_i = cc->cam_awareness[ca].calibration.position - u * -s;
							sdm3->min_dist_central_points_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = v_i - ((u_i - v_i) * -0.5f);
							*/
							//sdm3->distance_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = length(v_i - u_i);

							if (abs(target_t - t) < 1.0f && abs(target_s - s) < 1.0f
								//abs(width_t - width_s) > 0.3f 
								//sdm3->distance_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] > 0.5f
								) {
								sdm3->class_match_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = false;
							}

							//logger("dist", sdm3->distance_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i]);
						}

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
					}
					else {
						sdm3->class_match_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i] = false;
						//min dist of parallel lines
						//sdm3->matches[sm].distance[ca] = length(cross(v, pmp)) / length(v);
					}
				}
			}
		}
	}
}

void statistic_detection_matcher_3d_init(struct statistic_detection_matcher_3d* sdm3, int size, unsigned long long ttl, struct camera_control* cc, int population_c, struct statistic_camera_ray_data *scrd) {
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
	
	sdm3->scrd = scrd;
	/*
	sdm3->detections_3d = (struct statistic_detection_matcher_3d_detection*)malloc(cc->camera_count * sdm3->cdh_max_size * sizeof(struct statistic_detection_matcher_3d_detection));


	size_t matrix_base_size = cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size;

	sdm3->class_match_matrix				= (bool*)					malloc(matrix_base_size * sizeof(bool));

	sdm3->distance_matrix					= (float *)					malloc(matrix_base_size * sizeof(float));
	cudaMalloc(&sdm3->distance_matrix_device, matrix_base_size * sizeof(float));

	sdm3->min_dist_central_points_matrix	= (struct vector3<float> *) malloc(matrix_base_size * sizeof(struct vector3<float>));
	cudaMalloc(&sdm3->min_dist_central_points_matrix_device, matrix_base_size * sizeof(struct vector3<float>));

	sdm3->size_estimation_correction_dist_matrix				= (float *)					malloc(matrix_base_size * sizeof(float));
	cudaMalloc(&sdm3->size_estimation_correction_dist_matrix_device, matrix_base_size * sizeof(float));
		
	sdm3->ray_matrix = (struct vector3<float> *) malloc(cc->camera_count * (1 + sdm3->cdh_max_size) * sizeof(struct vector3<float>));
	cudaMalloc(&sdm3->ray_matrix_device, cc->camera_count * (1 + sdm3->cdh_max_size) * sizeof(struct vector3<float>));

	sdm3->single_ray_max_estimates = 5;
	cudaMalloc(&sdm3->single_ray_position_estimate_device, sdm3->single_ray_max_estimates * cc->camera_count * sdm3->cdh_max_size * sizeof(struct vector2<float>));
	*/

	sdm3->population_c = population_c;  //object count	+ object genetic
	size_t population_mem_size = sdm3->population_c * (size_t)(1 + (size_t)(size * cc->camera_count)) * sizeof(int);
	sdm3->population		= (int*) malloc(population_mem_size);
	sdm3->population_bak	= (int*) malloc(population_mem_size);
	cudaMalloc(&sdm3->population_device, population_mem_size);
	
	sdm3->population_scores = (float*)malloc(sdm3->population_c * sizeof(float));
	cudaMalloc(&sdm3->population_scores_device, sdm3->population_c * sizeof(float));

	sdm3->population_scores_idxs_orig = (int*)malloc(sdm3->population_c * sizeof(int));
	for (int i = 0; i < sdm3->population_c; i++) {
		sdm3->population_scores_idxs_orig[i] = i;
	}
	sdm3->population_scores_idxs = (int*)malloc(sdm3->population_c * sizeof(int));

	size_t population_evolution_buffer_mem_size = sdm3->population_c * (size_t)((size_t)(2 * sdm3->size) + (size_t)(2 * cc->camera_count * sdm3->cdh_max_size)) * sizeof(unsigned char);
	cudaMalloc(&sdm3->population_evolution_buffer_device, population_evolution_buffer_mem_size);

	sdm3->population_max_evolutions = 10;
	sdm3->population_keep_factor = 1.0f / 2.0f;
	sdm3->population_mutation_rate = 0.002f;

	sdm3->randoms_size = 1024;
	sdm3->randoms = (float * )malloc(sdm3->randoms_size * sizeof(float));
	cudaMalloc(&sdm3->randoms_device, sdm3->randoms_size * sizeof(float));

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		logger("CUDA Error: %s\n", cudaGetErrorString(err));
	}
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
		for (int o = 0; o < object_count[0]; o++) {
			bool empty_object = true;
			for (int r = 0; r < cc->camera_count; r++) {
				if (genetic_base_idx[o * cc->camera_count + r] != -1) {
					empty_object = false;
					break;
				}
			}
			if (empty_object) {
				if (o < object_count[0] - 1) {
					memcpy(&genetic_base_idx[o * cc->camera_count], &genetic_base_idx[(object_count[0] - 1) * cc->camera_count], cc->camera_count * sizeof(int));
					object_count[0]--;
					o--;
				} else {
					object_count[0]--;
				}
				if (object_count[0] < min_objects) {
					p--;
				}
			}
		}
		/*
		logger("p", p);
		logger("object_count", object_count[0]);
		for (int o = 0; o < object_count[0]; o++) {
			logger("o", o);
			for (int r = 0; r < cc->camera_count; r++) {
				logger(genetic_base_idx[o * cc->camera_count + r]);
			}
		}
		*/
	}
	
	free(rays_per_cam);
	free(rays_per_cam_wrk);

	free(init_member);
}

int iteration = 0;

void statistic_evolutionary_tracker_population_next_pool(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc) {
	float sum_scores = 0.0f;
	for (int p = 0; p < sdm3->population_c; p++) {
		if (sdm3->population_scores[p] >= 0.0f) {
			sum_scores += sdm3->population_scores[p];
		}
	}
	stable_sort(sdm3->population_scores_idxs, &sdm3->population_scores_idxs[sdm3->population_c], [sdm3](size_t i1, size_t i2) {return sdm3->population_scores[i1] < sdm3->population_scores[i2]; });
	//logger(sdm3->population_scores[sdm3->population_scores_idxs[0]]);
	
	//logger("iteration", iteration);
	for (int i = 0; i < sdm3->population_c; i++) {
		//logger(sdm3->population_scores_idxs[i]);
		//logger(sdm3->population_scores[sdm3->population_scores_idxs[i]]);
		//int *population_base = sdm3->pop
	}
	if (iteration == 1) {
		//exit(0);
	}
	iteration++;
	

	int *pop_cur = sdm3->population;
	int *pop_next = sdm3->population_bak;
	if (sdm3->population_swap) {
		pop_next = pop_cur;
		pop_cur = sdm3->population_bak;
	}

	int population_kept_c = (int)floorf(sdm3->population_c * sdm3->population_keep_factor);
	int total_new = 0;

	//logger("kept");

	for (int p = 0; p < population_kept_c; p++) {
		int target_idx = total_new * (1 + (sdm3->size * cc->camera_count));
		int origin_idx = sdm3->population_scores_idxs[p] * (1 + (sdm3->size * cc->camera_count));
		memcpy(&pop_next[target_idx], &pop_cur[origin_idx], (1 + (sdm3->size * cc->camera_count) * sizeof(int)));
		/*
		logger("p", p);
		logger("t_n", total_new);
		logger("object_count", pop_next[target_idx]);
		for (int o = 0; o < pop_next[target_idx]; o++) {
			logger("o", o);
			for (int r = 0; r < cc->camera_count; r++) {
				logger(pop_next[target_idx + 1 + o * cc->camera_count + r]);
			}
		}
		*/
		total_new++;
	}
	int tries = 0;
	float tries_factor = 1.0f;

	//logger("p new");

	//min score = best score 0.0 err on all triangulations, max score = worst score...
	float min_score = sdm3->population_scores[sdm3->population_scores_idxs[0]];
	float max_kept_score = sdm3->population_scores[sdm3->population_scores_idxs[population_kept_c - 1]];
	if (max_kept_score == min_score) max_kept_score += 1e-9;
	if (abs(max_kept_score - min_score) < 5.0f) {
		max_kept_score += 5.0f;
	}

	while (total_new < sdm3->population_c) {
		tries_factor = 1.0f * ((tries/(float)(sdm3->population_c - population_kept_c)) * 0.1f);
		int p_i = (int)(((rand() / (float)RAND_MAX)) * (sdm3->population_c * sdm3->population_keep_factor));
		float i_score = sdm3->population_scores[sdm3->population_scores_idxs[p_i]];
		float p_in = 1.0f - ((i_score - min_score) / (max_kept_score - min_score));
		float p_roll = (rand() / (float)RAND_MAX);
		if (p_roll <= p_in * tries_factor) {
			int target_idx = total_new * (1 + (sdm3->size * cc->camera_count));
			int origin_idx = sdm3->population_scores_idxs[p_i] * (1 + (sdm3->size * cc->camera_count));
			memcpy(&pop_next[target_idx], &pop_cur[origin_idx], (1 + (sdm3->size * cc->camera_count) * sizeof(int)));
			/*
			logger("p", total_new);
			logger("object_count", pop_next[target_idx]);
			for (int o = 0; o < pop_next[target_idx]; o++) {
				logger("o", o);
				for (int r = 0; r < cc->camera_count; r++) {
					logger(pop_next[target_idx + 1 + o * cc->camera_count + r]);
				}
			}
			*/
			total_new++;
		}
		tries++;
	}

	//logger("done");
}

void statistic_detection_matcher_3d_compute_evolutionary_result(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, struct camera_control_shared_state* ccss, int cdh_currents, int cdh_max_c, unsigned long long t_now, const int heatmap_gpu_id) {
	vector3<int> heatmap_dimensions = { cc->statistics_3d_in->heatmap_3d.sqg.dimensions[0], cc->statistics_3d_in->heatmap_3d.sqg.dimensions[1], cc->statistics_3d_in->heatmap_3d.sqg.dimensions[2] };
	vector3<float> heatmap_quantization_factors = { cc->statistics_3d_in->heatmap_3d.sqg.quantization_factors[0], cc->statistics_3d_in->heatmap_3d.sqg.quantization_factors[1], cc->statistics_3d_in->heatmap_3d.sqg.quantization_factors[2] };
	vector3<int> heatmap_span_start = { cc->statistics_3d_in->heatmap_3d.sqg.spans[0][0], cc->statistics_3d_in->heatmap_3d.sqg.spans[1][0], cc->statistics_3d_in->heatmap_3d.sqg.spans[2][0] };

	float* heatmap_device_ptr = (float*)(cc->statistics_3d_in->gmb->p_device + (heatmap_gpu_id * cc->statistics_3d_in->gmb->size));

	cudaMemcpyAsync(sdm3->scrd->distance_matrix_device, sdm3->scrd->distance_matrix, (size_t)cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaMemcpyAsync(sdm3->scrd->size_estimation_correction_dist_matrix_device, sdm3->scrd->size_estimation_correction_dist_matrix, (size_t)cc->camera_count * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);

	int max_objects = min(cdh_currents, sdm3->size);
	int min_objects = max((int)floorf(cdh_currents / (float)cc->camera_count), cdh_max_c);

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

		cudaMemcpyAsync(sdm3->population_device, pop_cur, (size_t)sdm3->population_c * (1 + (size_t)(sdm3->size * cc->camera_count)) * sizeof(int), cudaMemcpyHostToDevice, cuda_streams[0]);

		for (int ra = 0; ra < sdm3->randoms_size; ra++) {
			sdm3->randoms[ra] = (rand() / (float)RAND_MAX);
		}
		cudaMemcpyAsync(sdm3->randoms_device, sdm3->randoms, sdm3->randoms_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);

		memcpy(sdm3->population_scores_idxs, sdm3->population_scores_idxs_orig, sdm3->population_c * sizeof(int));

		cudaStreamSynchronize(cuda_streams[0]);

		if (e > 0) {
			cudaMemsetAsync(sdm3->population_evolution_buffer_device, 0, sdm3->population_c * (2 * sdm3->size + 2 * cc->camera_count * sdm3->cdh_max_size) * sizeof(unsigned char), cuda_streams[3]);
			statistics_evolutionary_tracker_population_evolve_kernel_launch(sdm3->size, cc->camera_count, sdm3->cdh_max_size, sdm3->population_device, sdm3->population_evolution_buffer_device, sdm3->population_scores_device, sdm3->population_c, sdm3->population_keep_factor, sdm3->population_mutation_rate, sdm3->randoms_device, sdm3->randoms_size, min_objects, max_objects);
			cudaMemcpyAsync(pop_cur, sdm3->population_device, sdm3->population_c * (1 + (sdm3->size * cc->camera_count)) * sizeof(int), cudaMemcpyDeviceToHost, cuda_streams[4]);
		}

		statistics_evolutionary_tracker_kernel_launch(sdm3->scrd->distance_matrix_device, sdm3->scrd->min_dist_central_points_matrix_device, sdm3->size, cc->camera_count, sdm3->cdh_max_size, sdm3->population_device, sdm3->population_scores_device, sdm3->population_c, sdm3->scrd->single_ray_position_estimate_device, sdm3->scrd->single_ray_max_estimates, heatmap_dimensions, heatmap_quantization_factors, heatmap_span_start, heatmap_device_ptr);
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

	int tmp_o = 0;
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
				sdm3->detections[tmp_o].class_id = 37;
				sdm3->detections[tmp_o].velocity = { 0.0f, 0.0f, 0.0f };

				sdm3->detections[tmp_o].score = 0.0f;
				sdm3->detections[tmp_o].timestamp = t_now;
				if (ccss != nullptr && shared_objects < cc->camera_count * 5) {
					ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = sdm3->detections[tmp_o++].position;
					shared_objects++;
				}
				if (tmp_o == sdm3->size) break;

				if (c_id_last > -1) {
					sdm3->detections[o].position = sdm3->detections[o].position - -sdm3->scrd->min_dist_central_points_matrix[c_id_last * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + r_id_last * cc->camera_count * sdm3->cdh_max_size + c_id * sdm3->cdh_max_size + r_id];
					cc->cam_awareness[c_id].calibration.position;
					sdm3->scrd->detections_3d[c_id * sdm3->cdh_max_size + r_id].direction;
					ray_count++;
				}
				c_id_last = c_id;
				r_id_last = r_id;
			}
		}
		if (tmp_o == sdm3->size) break;

		if (ray_count > 0) {
			sdm3->detections[o].position = sdm3->detections[o].position / (float)ray_count;

			if (ccss != nullptr && shared_objects < cc->camera_count * 5) {
				ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = sdm3->detections[o].position;
				shared_objects++;
			}
		}

	}
}

void statistic_detection_matcher_3d_get_position_for_max_objects_1(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc) {
	memset(sdm3->is_matched, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
	memset(sdm3->detections_buffer, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(struct cam_detection_3d));

	bool* used_current = (bool*)malloc(cc->camera_count * sdm3->cdh_max_size);

	int used_count = 0;
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		for (int c = 0; c < cdh->latest_count; c++) {
			struct statistic_detection_matcher_3d_detection* sdm3dd = &sdm3->scrd->detections_3d[ca * sdm3->cdh_max_size + c];
			if (sdm3->is_matched[ca * sdm3->cdh_max_size + c]) continue;
			for (int ca_i = 0; ca_i < cc->camera_count; ca_i++) {
				if (ca_i == ca) continue;
				struct cam_detection_history* cdh_i = &cc->cam_awareness[ca_i].detection_history;

				for (int c_i = 0; c_i < cdh_i->latest_count; c_i++) {
					if (!sdm3->is_matched[ca_i * sdm3->cdh_max_size + c_i] &&
						sdm3->scrd->class_match_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i]) {

						used_current[ca * sdm3->cdh_max_size + c] = true;
						used_current[ca_i * sdm3->cdh_max_size + c_i] = true;
						sdm3->detections_buffer[0].timestamp = sdm3dd->timestamp;
						/*
						logger("coord");
						logger(sdm3->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + best_score_idx][0]);
						logger(sdm3->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + best_score_idx][1]);
						logger(sdm3->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + best_score_idx][2]);
						*/
						if (used_count > 0 && length((sdm3->detections_buffer[0].position - -sdm3->scrd->min_dist_central_points_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i])) > 0.5f) {
							sdm3->detections_buffer[0].position = (sdm3->detections_buffer[0].position - -sdm3->scrd->min_dist_central_points_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i]);
							used_count++;
						}
						else if (used_count == 0) {
							sdm3->detections_buffer[0].position = (sdm3->detections_buffer[0].position - -sdm3->scrd->min_dist_central_points_matrix[ca * (sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size) + c * (cc->camera_count * sdm3->cdh_max_size) + ca_i * sdm3->cdh_max_size + c_i]);
							used_count++;
						}
					}
				}
			}
		}
	}
	if (used_count > 0) {
		sdm3->detections_buffer[0].class_id = 37;
		sdm3->detections_buffer[0].position = sdm3->detections_buffer[0].position * (1.0f / (float)used_count);
		sdm3->detections_buffer[0].score = 0.0f;
	}
}

struct vector2<int> statistic_detection_matcher_3d_greedy_ray_groups(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc) {
	memset(sdm3->is_matched, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(bool));
	memset(sdm3->detections_buffer, 0, cc->camera_count * sdm3->cdh_max_size * sizeof(struct cam_detection_3d));

	bool* used_current = (bool*)malloc(cc->camera_count * sdm3->cdh_max_size);

	//assign ray groups
	float best_inv_score = 10000.0f;
	int ca_best = -1;
	int c_best = -1;
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		for (int c = 0; c < cdh->latest_count; c++) {
			if (sdm3->is_matched[ca * sdm3->cdh_max_size + c]) continue;
			float inv_best_score = 0.0f;
			memset(used_current, 0, sizeof(cc->camera_count * sdm3->cdh_max_size));
			used_current[ca * sdm3->cdh_max_size + c] = true;
			int used_count = 0;
			struct statistic_detection_matcher_3d_detection* sdm3dd = &sdm3->scrd->detections_3d[ca * sdm3->cdh_max_size + c];
			sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].ray_position[used_count] = cc->cam_awareness[ca].calibration.position;
			sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].ray_direction[used_count] = sdm3dd->direction;
			for (int ca_i = 0; ca_i < cc->camera_count; ca_i++) {
				if (ca_i == ca) continue;
				struct cam_detection_history* cdh_i = &cc->cam_awareness[ca_i].detection_history;

				int best_score_idx = -1;
				float best_dist = 1000000.0f;

				for (int c_i = 0; c_i < cdh_i->latest_count; c_i++) {
					if (!sdm3->is_matched[ca_i * sdm3->cdh_max_size + ca_i]) {
						if (sdm3->scrd->class_match_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i]) {
							if (sdm3->scrd->distance_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i] < best_dist) {
								best_score_idx = c_i;
								best_dist = sdm3->scrd->distance_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + c_i];
							}
						}
					}
				}
				if (best_score_idx > -1 && best_dist < 0.5f) {
					float some_threshold = 5.0f;
					if (best_dist < some_threshold) {
						used_current[ca_i * sdm3->cdh_max_size + best_score_idx] = true;
						//TMP
						inv_best_score += best_dist;
						used_count++;
						if (used_count < 5) {
							sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].ray_position[used_count] = cc->cam_awareness[ca_i].calibration.position;
							sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].ray_direction[used_count] = sdm3->scrd->detections_3d[ca_i * sdm3->cdh_max_size + best_score_idx].direction;
						}
						sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].position = (sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].position - -sdm3->scrd->min_dist_central_points_matrix[ca * sdm3->cdh_max_size * cc->camera_count * sdm3->cdh_max_size + ca_i * sdm3->cdh_max_size + best_score_idx]);
					}
				}
			}
			if (used_count > 0) {
				if (inv_best_score < best_inv_score) {
					best_inv_score = inv_best_score;
					ca_best = ca;
					c_best = c;
				}
				//logger("inv_best_score", inv_best_score);
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
	return struct vector2<int>(ca_best, c_best);
}

void statistic_detection_matcher_3d_remove_all_but_best(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, int ca_best, int c_best) {
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		for (int c = 0; c < cdh->latest_count; c++) {
			if (ca != ca_best || c != c_best) {
				sdm3->detections_buffer[ca * sdm3->cdh_max_size + c].timestamp = 0;
			}
		}
	}
}

void statistic_detection_matcher_3d_calculate_ground_projection(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, struct camera_control_shared_state* ccss) {
	int o = 0;
	int shared_objects = 0;
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		for (int c = 0; c < cdh->latest_count; c++) {
			sdm3->detections[o].class_id = 37;
			sdm3->detections[o].velocity = { 0.0f, 0.0f, 0.0f };

			//logger(sdm3->detections_3d[ca * sdm3->cdh_max_size + c].direction[2]);
			if (sdm3->scrd->detections_3d[ca * sdm3->cdh_max_size + c].direction[2] != 0) {
				float l = -cc->cam_awareness[ca].calibration.position[2] / sdm3->scrd->detections_3d[ca * sdm3->cdh_max_size + c].direction[2];
				sdm3->detections[o].position = {
					cc->cam_awareness[ca].calibration.position[0] + l * sdm3->scrd->detections_3d[ca * sdm3->cdh_max_size + c].direction[0],
					cc->cam_awareness[ca].calibration.position[1] + l * sdm3->scrd->detections_3d[ca * sdm3->cdh_max_size + c].direction[1],
					0.0f
				};
				sdm3->detections[o].timestamp = 1;
				if (ccss != nullptr && shared_objects < cc->camera_count * 5) {
					ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = sdm3->detections[o].position;
					shared_objects++;
				}
				o++;
			}
			if (o == sdm3->size) break;
		}
		if (o == sdm3->size) break;
	}
}

void statistic_detection_matcher_3d_greedy_tracker_matching(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, struct camera_control_shared_state* ccss) {
	int shared_objects = 0;
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
				/*
				logger(d);
				logger(sdm3->detections[d].position[0]);
				logger(sdm3->detections[d].position[1]);
				logger(sdm3->detections[d].position[2]);
				logger(sdm3->detections[d].timestamp);
				*/
				if (ccss != nullptr && shared_objects < cc->camera_count * 5) {
					ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = sdm3->detections[d].position;
					memcpy(&ccss[shared_objects / 5].latest_detections_3d[shared_objects % 5], &sdm3->detections[d], sizeof(struct cam_detection_3d));
					shared_objects++;
				}
			}
		}
		else {
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
				//logger("adding new");
				if (ccss != nullptr && shared_objects < cc->camera_count * 5) {
					ccss[shared_objects / 5].latest_detections_objects[shared_objects % 5] = sdm3->detections[d].position;
					shared_objects++;
				}

				break;
			}
		}
	}
}

struct vector2<int> statistic_detection_matcher_3d_get_camera_detection_meta(struct camera_control* cc) {
	int cdh_max_c = 0;
	int cdh_currents = 0;
	for (int ca = 0; ca < cc->camera_count; ca++) {
		struct cam_detection_history* cdh = &cc->cam_awareness[ca].detection_history;
		cdh_currents += cdh->latest_count;
		if (cdh->latest_count > cdh_max_c) {
			cdh_max_c = cdh->latest_count;
		}
	}

	return struct vector2<int>(cdh_max_c, cdh_currents);
}

void statistic_detection_matcher_3d_update(struct statistic_detection_matcher_3d* sdm3, struct camera_control* cc, struct camera_control_shared_state *ccss) {

	struct vector2<int> camera_det_meta = statistic_detection_matcher_3d_get_camera_detection_meta(cc);
	int cdh_max_c = camera_det_meta[0];
	int cdh_currents = camera_det_meta[1];
	
	if (cdh_currents == 0) return;

	//precompute ray meta
	unsigned long long t_now = statistic_camera_ray_data_convert_to_3d_detection(sdm3->scrd, cc, ccss);
	
	//int heatmap_gpu_id = statistic_camera_ray_data_prepare_position_estimation_from_heatmap_async(sdm3->scrd, cc, 2);

	statistic_camera_ray_data_precompute_ray_correlation(sdm3->scrd, cc);

	//[e]
	//cudaMemcpyAsync(sdm3->scrd->min_dist_central_points_matrix_device, sdm3->scrd->min_dist_central_points_matrix, (size_t)cc->camera_count* sdm3->cdh_max_size* cc->camera_count* sdm3->cdh_max_size * sizeof(struct vector3<float>), cudaMemcpyHostToDevice, cuda_streams[0]);

	//statistic_camera_ray_data_wait_for_position_estimation_from_heatmap(cc, heatmap_gpu_id, 2);
	
	//[e]
	//statistic_detection_matcher_3d_compute_evolutionary_result(sdm3, cc, ccss, cdh_currents, cdh_max_c, t_now, heatmap_gpu_id);

	//[1]
	//statistic_detection_matcher_3d_get_position_for_max_objects_1(sdm3, cc);

	//[!1]
	struct vector2<int> best_det = statistic_detection_matcher_3d_greedy_ray_groups(sdm3, cc);
	statistic_detection_matcher_3d_remove_all_but_best(sdm3, cc, best_det[0], best_det[1]);
	
	
	memset(sdm3->is_final_matched, 0, cc->camera_count* sdm3->cdh_max_size * sizeof(bool));
	if (ccss != nullptr) {
		for (int ca = 0; ca < cc->camera_count; ca++) {
			memset(&ccss[ca].latest_detections_objects, 0, 5 * sizeof(struct vector3<float>));
			memset(&ccss[ca].latest_detections_3d, 0, 5 * sizeof(struct cam_detection_3d));
		}
	}
	
	//statistic_detection_matcher_3d_calculate_ground_projection(sdm3, cc, ccss);
	
	statistic_detection_matcher_3d_greedy_tracker_matching(sdm3, cc, ccss);
	
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

	int idx = 0; 
	while (idx < sv3d->sqg.total_size / sv3d->sqg.data_size) {
		sv3d->data[idx] = 1.0f / 27.0f;
		sv3d->data[idx + 1] = 0;
		sv3d->data[idx + 2] = 0;
		idx += 3;
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

		//TMP
		sv3d->data[idx + idx_inner + 1] = ((99.0f / 100.0f) * sv3d->data[idx + idx_inner + 1] + (1.0f / 100.0f) * velocity_t);
		if (sv3d->data[idx + idx_inner + 1] > sv3d->max_vel) sv3d->max_vel = sv3d->data[idx + idx_inner + 1];

		sv3d->data[idx + idx_inner + 2] = ((99.0f / 100.0f) * sv3d->data[idx + idx_inner + 2] + (1.0f / 100.0f) * acceleration_t);
		if (sv3d->data[idx + idx_inner + 2] > sv3d->max_acc) sv3d->max_acc = sv3d->data[idx + idx_inner + 2];
	} else {
		logger("idx -1");
		logger(position[0]);
		logger(position[1]);
		logger(position[2]);
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

void statistic_unscatter_interpolation_init(struct statistic_unscatter_interpolation_2d* sui2d, struct vector2<int> grid_size, struct vector2<int> dimension) {
	sui2d->grid_size = grid_size;
	sui2d->dimension = dimension;

	sui2d->data = (float*)malloc(grid_size[0] * grid_size[1] * sizeof(float));
}

float statistic_unscatter_orth_proj(struct vector2<float> p1, struct vector2<float> p2, struct vector2<float> s, float* out_dist);

void statistic_unscatter_interpolation_calculate(struct statistic_unscatter_interpolation_2d* sui2d, std::vector<struct vector2<float>> points, std::vector<float> values, int power) {
	float x_steps = sui2d->dimension[0] / (float)sui2d->grid_size[0];
	float x_start = x_steps / 2.0f;

	float y_steps = sui2d->dimension[1] / (float)sui2d->grid_size[1];
	float y_start = y_steps / 2.0f;

	for (int r = 0; r < sui2d->grid_size[1]; r++) {
		for (int c = 0; c < sui2d->grid_size[0]; c++) {
			sui2d->data[r * sui2d->grid_size[0] + c] = 0.0f;

			struct vector2<float> target = { x_start + c * x_steps, y_start + r * y_steps };

			float d = 0.0f;
			bool hit = false;

			int max_id = 0;
			float max_dist = (float)max(sui2d->dimension[0], sui2d->dimension[1]);
			float closest_dists[4] = { max_dist, max_dist, max_dist, max_dist };
			int closest_ids[4] = { -1, -1, -1, -1 };

			for (int m = 0; m < 4; m++) {
				for (int p = 0; p < points.size(); p++) {
					float dist = length(target - points[p]);
					if (dist < 0.1f) {
						sui2d->data[r * sui2d->grid_size[0] + c] = values[p];
						hit = true;
						break;
					} else {
						if (closest_ids[m] == -1) {
							if (m == 0 || (m > 0 && (closest_dists[m - 1] < dist || (closest_ids[m - 1] != p && closest_dists[m - 1] == dist)))) {
								closest_ids[m] = p;
								closest_dists[m] = dist;
							}
						} else {
							if (closest_dists[m] > dist) {
								if (m == 0 || (m > 0 && (closest_dists[m - 1] < dist || (closest_ids[m - 1] != p && closest_dists[m - 1] == dist)))) {
									closest_ids[m] = p;
									closest_dists[m] = dist;
								}
							}
						}
					}
				}
				if (hit) break;
			}
			if (!hit) {
				//point furthest away
				struct vector2<float> p_1 = points[closest_ids[3]];
				
				//point with least x dist from p_1
				struct vector2<float> p_2 = points[closest_ids[2]];
				int					  p_2_id = 2;
				float tmp_dist_x = abs(p_1[0] - p_2[0]);
				for (int m = 0; m < 2; m++) {
					float d = abs(p_1[0] - points[closest_ids[m]][0]);
					if (d < tmp_dist_x) {
						tmp_dist_x = d;
						p_2_id = m;
					}
				}
				p_2 = points[closest_ids[p_2_id]];

				//point with least y dist from p_1
				struct vector2<float> p_3;
				int					  p_3_id;
				float tmp_dist_y = (float)sui2d->dimension[1];
				for (int m = 0; m < 3; m++) {
					if (m != p_2_id) {
						float d = abs(p_1[1] - points[closest_ids[m]][1]);
						if (d < tmp_dist_y) {
							tmp_dist_y = d;
							p_3_id = m;
						}
					}
				}
				p_3 = points[closest_ids[p_3_id]];

				//last point
				struct vector2<float> p_4;
				int					  p_4_id;
				for (int m = 0; m < 3; m++) {
					if (m != p_2_id && m != p_3_id) {
						p_4 = points[closest_ids[m]];
						p_4_id = m;
						break;
					}
				}

				float v_p1 = values[closest_dists[3]];
				float v_p2 = values[closest_dists[p_2_id]];
				float v_p3 = values[closest_dists[p_3_id]];
				float v_p4 = values[closest_dists[p_4_id]];

				float d_x1		= 0.0f;
				float lambda	= statistic_unscatter_orth_proj(p_1, p_3, target, &d_x1);
				float v_x1		= v_p1 + lambda * (v_p3 - v_p1);

				float d_x2		= 0.0f;
				float phi		= statistic_unscatter_orth_proj(p_2, p_4, target, &d_x2);
				float v_x2		= v_p2 + phi * (v_p4 - v_p2);

				float d_y1		= 0.0f;
				float xi		= statistic_unscatter_orth_proj(p_1, p_2, target, &d_y1);
				float v_y1		= v_p1 + xi * (v_p2 - v_p1);

				float d_y2		= 0.0f;
				float theta		= statistic_unscatter_orth_proj(p_3, p_4, target, &d_y2);
				float v_y2		= v_p3 + theta * (v_p4 - v_p3);

				float d_total	= d_x1 + d_x2 + d_y1 + d_y2;

				float p__x1 = (float)pow(1.0f - (d_x1 / d_total), power);
				float p__x2 = (float)pow(1.0f - (d_x2 / d_total), power);
				float p__y1 = (float)pow(1.0f - (d_y1 / d_total), power);
				float p__y2 = (float)pow(1.0f - (d_y2 / d_total), power);

				float p_total = p__x1 + p__x2 + p__y1 + p__y2;

				p__x1 /= p_total;
				p__x2 /= p_total;
				p__y1 /= p_total;
				p__y2 /= p_total;

				float value = p__x1 * v_x1 + p__x2 * v_x2 + p__y1 * v_y1 + p__y2 * v_y2;
				sui2d->data[r * sui2d->grid_size[0] + c] = value;
			}
		}
	}
}

float statistic_unscatter_orth_proj(struct vector2<float> p1, struct vector2<float> p2, struct vector2<float> s, float* out_dist) {
	struct vector2<float> n = { -(p2[1] - p1[1]), (p2[0] - p1[0]) };
	
	float lambda = 0.0f;
	if (n[0] == 0) {
		lambda = (s[0] - p1[0]) / (p2[0] - p1[0]);
	} else if (n[1] == 0) {
		lambda = (s[1] - p1[1]) / (p2[1] - p1[1]);
	} else {
		float lambda_n = (p1[0] / n[0]) - (s[0] / n[0]) - (p1[1] / n[1]) + (s[1] / n[1]);
		float lambda_d = ((p2[1] - p1[1]) / n[1]) - ((p2[0] - p1[0]) / n[0]);

		lambda = lambda_n / lambda_d;
	}

	struct vector2<float> p = p1 - -(p2 - p1) * lambda;
	float d = length(s - p);
	*out_dist = d;
	return lambda;
}

void statistic_unscatter_interpolation_destroy(struct statistic_unscatter_interpolation_2d* sui2d) {
	free(sui2d->data);
}

void statistic_unscatter_triangulation_init(struct statistic_unscatter_triangulation_2d* sut2d, struct vector2<int> grid_size, struct vector2<int> dimension) {
	sut2d->grid_size = grid_size;
	sut2d->dimension = dimension;

	sut2d->data = (float*)malloc(grid_size[0] * grid_size[1] * sizeof(float));
}

bool statistic_unscatter_triangulation_is_in_triangle(struct vector2<float> p0, struct vector2<float> p1, struct vector2<float> p2, struct vector2<float> point, float *out_A, float* out_s, float* out_ts) {
	struct vector2<float> point_t = point - p0;

	struct vector2<float> u = p1 - p0;
	struct vector2<float> v = p2 - p0;

	struct vector2<float> w_h = p2 - p1;

	float u_l = length(u * 0.1f);
	float v_l = length(v * 0.1f);
	float w_l = length(w_h * 0.1f);

	float s_h = (u_l + v_l + w_l) * 0.5f;

	float A_1 = s_h - u_l;
	float A_2 = s_h - v_l;
	float A_3 = s_h - w_l;

	A_1 *= (A_1 > 0);
	A_2 *= (A_1 > 0);
	A_3 *= (A_1 > 0);

	A_1 = sqrtf(A_1);
	A_2 = sqrtf(A_2);
	A_3 = sqrtf(A_3);

	float A = sqrtf(s_h) * A_1 * A_2 * A_3;
	if (A == 0) return false;

	float uu = dot(u, u);
	float uv = dot(u, v);
	float vv = dot(v, v);

	float wu = dot(point_t, u);
	float wv = dot(point_t, v);

	float D = uv * uv - uu * vv;

	if (D == 0) return false;

	float s = (uv * wv - vv * wu) / D;
	if (s < 0.0f || s > 1.0f) return false;
	float ts = (uv * wu - uu * wv) / D;
	if (ts < 0.0f || s + ts > 1.0f) return false;
	*out_A = A;
	*out_s = s;
	*out_ts = ts;
	return true;
}

// triangulation video dump
/*
VideoWriter outputVideo;
int total_frames = 600;
int frame_count = 0;
*/

void statistic_unscatter_triangulation_calculate(struct statistic_unscatter_triangulation_2d* sut2d, std::vector<struct vector2<float>> points, std::vector<float> values) {
	float x_steps = sut2d->dimension[0] / (float)sut2d->grid_size[0];
	float x_start = x_steps / 2.0f;

	float y_steps = sut2d->dimension[1] / (float)sut2d->grid_size[1];
	float y_start = y_steps / 2.0f;

	for (int r = 0; r < sut2d->grid_size[1]; r++) {
		for (int c = 0; c < sut2d->grid_size[0]; c++) {
			sut2d->data[r * sut2d->grid_size[0] + c] = 0.0f;

			struct vector2<float> target = { x_start + c * x_steps, y_start + r * y_steps };

			bool hit = false;
			float closest_dist = sut2d->dimension[0];
			int closest_id = -1;
			for (int p = 0; p < points.size(); p++) {
				float dist = length(points[p] - target);

				if (dist < 0.1) {
					sut2d->data[r * sut2d->grid_size[0] + c] = values[p];
					hit = true;
					break;
				} else {
					if (dist < closest_dist) {
						closest_dist = dist;
						closest_id = p;
					}
				}
			}
			if (!hit) {
				float smallest_A = sut2d->dimension[0] * sut2d->dimension[1];
				for (int p1 = 0; p1 < points.size(); p1++) {
					if (p1 != closest_id) {
						for (int p2 = 0; p2 < points.size(); p2++) {
							if (p2 != p1 && p2 != closest_id) {
								float A = 0.0f;
								float s = 0.0f;
								float ts = 0.0f;
								if (statistic_unscatter_triangulation_is_in_triangle(points[closest_id], points[p1], points[p2], target, &A, &s, &ts)) {
									float p_d_len_threshold = max((sut2d->dimension[0] / sut2d->grid_size[0]) * 1.5f * sqrtf(2.0f), (sut2d->dimension[1] / sut2d->grid_size[1]) * 1.5f * sqrtf(2.0f));
									if (A < smallest_A && max(length(points[closest_id] - points[p1]), max(length(points[closest_id] - points[p2]), length(points[p1] - points[p2]))) < p_d_len_threshold) {
										smallest_A = A;
										sut2d->data[r * sut2d->grid_size[0] + c] = values[closest_id] + s * (values[p1] - values[closest_id]) + ts * (values[p2] - values[closest_id]);

										// triangulation video dump
										/*
										if (frame_count < total_frames) {
											if (length(points[closest_id] - points[p1]) < 200 && length(points[closest_id] - points[p2]) < 200 && length(points[p1] - points[p2]) < 200) {
												if (!outputVideo.isOpened()) {
													outputVideo.open("R:\\Cams\\triangulation.avi", cv::VideoWriter::fourcc('D', 'I', 'V', '5'), 30.0, cv::Size(640, 360), true);
												}
												Mat scattered = cv::Mat(360, 640, CV_8UC3);
												memset(scattered.data, 0, 360 * 640 * 3);

												cv::line(scattered, cv::Point(points[closest_id][0], points[closest_id][1]), cv::Point(points[p1][0], points[p1][1]), cv::Scalar(255, 255, 255), 2);
												cv::line(scattered, cv::Point(points[closest_id][0], points[closest_id][1]), cv::Point(points[p2][0], points[p2][1]), cv::Scalar(255, 255, 255), 2);
												cv::line(scattered, cv::Point(points[p1][0], points[p1][1]), cv::Point(points[p2][0], points[p2][1]), cv::Scalar(255, 255, 255), 2);

												cv::circle(scattered, cv::Point(target[0], target[1]), 3, cv::Scalar(0, 255, 0));

												outputVideo.write(scattered);
												frame_count++;
											}
										} else {
											if (outputVideo.isOpened()) {
												outputVideo.release();
											}
										}
										*/									

										hit = true;
									}
								}
							}
						}
					}
				}
				if (!hit) sut2d->data[r * sut2d->grid_size[0] + c] = FLT_MAX;
			}
		}
	}

}

void statistic_unscatter_triangulation_destroy(struct statistic_unscatter_triangulation_2d* sut2d) {
	free(sut2d->data);
}

bool statistic_unscatter_triangulation_get_value(struct statistic_unscatter_triangulation_2d* sut2d, struct vector2<float> point, float *out_value) {
	float x_steps = sut2d->dimension[0] / (float)sut2d->grid_size[0];
	float x_start = x_steps / 2.0f;

	float y_steps = sut2d->dimension[1] / (float)sut2d->grid_size[1];
	float y_start = y_steps / 2.0f;

	int x_index = floorf((point[0]-x_start) / x_steps);
	int y_index = floorf((point[1]-y_start) / y_steps);

	if (x_index >= 0 && x_index < sut2d->grid_size[0] - 1) {
		if (y_index >= 0 && y_index < sut2d->grid_size[1] - 1) {
			if (sut2d->data[y_index * sut2d->grid_size[0] + x_index] == FLT_MAX || sut2d->data[y_index * sut2d->grid_size[0] + x_index + 1] == FLT_MAX ||
				sut2d->data[(y_index + 1) * sut2d->grid_size[0] + x_index] == FLT_MAX || sut2d->data[(y_index + 1) * sut2d->grid_size[0] + x_index + 1] == FLT_MAX) {
				return false;
			}

			float x1 = x_start + x_index		* x_steps;
			float x2 = x_start + (x_index + 1)	* x_steps;

			float lambda = (point[0] - x1) / (x2 - x1);

			float y1 = y_start + y_index		* y_steps;
			float y2 = y_start + (y_index + 1)	* y_steps;

			float phi = (point[1] - y1) / (y2 - y1);

			float value_y1_x = sut2d->data[y_index * sut2d->grid_size[0] + x_index] + lambda * (sut2d->data[y_index * sut2d->grid_size[0] + x_index + 1] - sut2d->data[y_index * sut2d->grid_size[0] + x_index]);
			float value_y2_x = sut2d->data[(y_index + 1) * sut2d->grid_size[0] + x_index] + lambda * (sut2d->data[(y_index + 1) * sut2d->grid_size[0] + x_index + 1] - sut2d->data[(y_index + 1) * sut2d->grid_size[0] + x_index]);

			float value = value_y1_x + phi * (value_y2_x - value_y1_x);
			*out_value = value;

			return true;
		}
	}
	return false;
}

float statistic_unscatter_triangulation_center_shift_inverse(struct statistic_unscatter_triangulation_2d* sut2d) {
	struct vector2<float> point = { sut2d->dimension[0] * 0.5f, sut2d->dimension[1] * 0.5f };
	float value = 0.0f;
	if (statistic_unscatter_triangulation_get_value(sut2d, point, &value)) {
		for (int i = 0; i < sut2d->grid_size[0] * sut2d->grid_size[1]; i++) {
			if (sut2d->data[i] != FLT_MAX) {
				sut2d->data[i] -= value;
				sut2d->data[i] *= -1;
			}
		}
	}
	return value;
}

bool statistic_unscatter_triangulation_approximate_missing(struct statistic_unscatter_triangulation_2d* sut2d) {
	bool no_missing_values = true;
	for (int i = 0; i < sut2d->grid_size[0] * sut2d->grid_size[1]; i++) {
		if (sut2d->data[i] == FLT_MAX) {
			int row = i / sut2d->grid_size[0];
			int col = i % sut2d->grid_size[0];
			if (col - 1 >= 0 && col + 1 < sut2d->grid_size[0] && sut2d->data[row * sut2d->grid_size[0] + col - 1] != FLT_MAX && sut2d->data[row * sut2d->grid_size[0] + col + 1] != FLT_MAX) {
				sut2d->data[i] = (sut2d->data[row * sut2d->grid_size[0] + col - 1] + sut2d->data[row * sut2d->grid_size[0] + col + 1]) * 0.5f;
			} else if (col + 2 < sut2d->grid_size[0] && sut2d->data[row * sut2d->grid_size[0] + col + 1] != FLT_MAX && sut2d->data[row * sut2d->grid_size[0] + col + 2] != FLT_MAX) {
				sut2d->data[i] = sut2d->data[row * sut2d->grid_size[0] + col + 2] + 2.0f * (sut2d->data[row * sut2d->grid_size[0] + col + 1] - sut2d->data[row * sut2d->grid_size[0] + col + 2]);
				
			} else if (col - 2 >= 0 && sut2d->data[row * sut2d->grid_size[0] + col - 1] != FLT_MAX && sut2d->data[row * sut2d->grid_size[0] + col - 2] != FLT_MAX) {
				sut2d->data[i] = sut2d->data[row * sut2d->grid_size[0] + col - 2] + 2.0f * (sut2d->data[row * sut2d->grid_size[0] + col - 1] - sut2d->data[row * sut2d->grid_size[0] + col - 2]);
			}
			if (row - 1 >= 0 && row + 1 < sut2d->grid_size[1] && sut2d->data[(row - 1) * sut2d->grid_size[0] + col] != FLT_MAX && sut2d->data[(row + 1) * sut2d->grid_size[0] + col] != FLT_MAX) {
				float val = (sut2d->data[(row - 1) * sut2d->grid_size[0] + col] + sut2d->data[(row + 1) * sut2d->grid_size[0] + col]) * 0.5f;
				if (sut2d->data[i] != FLT_MAX) {
					sut2d->data[i] = (sut2d->data[i] + val) * 0.5f;
				} else {
					sut2d->data[i] = val;
				}
			} else if (row + 2 < sut2d->grid_size[1] && sut2d->data[(row + 1) * sut2d->grid_size[0] + col] != FLT_MAX && sut2d->data[(row + 2) * sut2d->grid_size[0] + col] != FLT_MAX) {
				float val = sut2d->data[(row + 2) * sut2d->grid_size[0] + col] + 2.0f * (sut2d->data[(row + 1) * sut2d->grid_size[0] + col] - sut2d->data[(row + 2) * sut2d->grid_size[0] + col]);
				if (sut2d->data[i] != FLT_MAX) {
					sut2d->data[i] = (sut2d->data[i] + val) * 0.5f;
				} else {
					sut2d->data[i] = val;
				}
			} else if (row - 2 >= 0 && sut2d->data[(row - 1) * sut2d->grid_size[0] + col] != FLT_MAX && sut2d->data[(row - 2) * sut2d->grid_size[0] + col] != FLT_MAX) {
				float val = sut2d->data[(row - 2) * sut2d->grid_size[0] + col] + 2.0f * (sut2d->data[(row - 1) * sut2d->grid_size[0] + col] - sut2d->data[(row - 2) * sut2d->grid_size[0] + col]);
				if (sut2d->data[i] != FLT_MAX) {
					sut2d->data[i] = (sut2d->data[i] + val) * 0.5;
				} else {
					sut2d->data[i] = val;
				}
			}
			if (sut2d->data[i] == FLT_MAX) {
				no_missing_values = false;
			}
		}
	}
	return no_missing_values;
}

void statistic_position_regression_init(struct statistic_position_regression* spr, struct vector3<float> parameter_search_space, struct camera_control* cc, string temporary_storage_dir, struct statistic_camera_ray_data *scrd, int t_samples_count) {
	spr->t_current = 0;
	spr->t_c = 0;

	spr->parameter_search_space = parameter_search_space;
	spr->camera_c = cc->camera_count;
	spr->stepsize = 0.1f;

	spr->camera_positions = (struct vector3<float> *) malloc(spr->camera_c * sizeof(struct vector3<float>));
	spr->camera_fov_factors = (float*)malloc(spr->camera_c * sizeof(float));
	spr->camera_resolutions_x = (int*)malloc(spr->camera_c * sizeof(int));
	for (int c = 0; c < spr->camera_c; c++) {
		spr->camera_positions[c] = cc->cam_awareness[c].calibration.position;
		spr->camera_fov_factors[c] = tan(cc->cam_awareness[c].calibration.lens_fov[0] * 0.5f * (M_PI / (180.0f)));
		spr->camera_resolutions_x[c] = cc->cam_meta[c].resolution[0];
	}

	spr->temporary_storage_dir = temporary_storage_dir;
	spr->t_samples_count = t_samples_count;

	spr->scrd = scrd;

	spr->parallel_c = 2048 * 20;
}

void statistic_position_regression_update(struct statistic_position_regression* spr, struct camera_control *cc) {
	unsigned long long t_now = statistic_camera_ray_data_convert_to_3d_detection(spr->scrd, cc, nullptr);

	if (t_now != spr->t_current) {
		statistic_camera_ray_data_precompute_ray_correlation(spr->scrd, cc);

		size_t matrix_base_size = cc->camera_count * spr->scrd->cdh_max_size * cc->camera_count * spr->scrd->cdh_max_size;

		stringstream ss_filename;
		ss_filename << spr->temporary_storage_dir << spr->t_c << "_";

		stringstream ss_filename_det_3d;
		ss_filename_det_3d << ss_filename.str() << "detections_3d.bin";
		util_write_binary(ss_filename_det_3d.str(), (unsigned char *) spr->scrd->detections_3d, spr->camera_c * spr->scrd->cdh_max_size * sizeof(struct statistic_detection_matcher_3d_detection));
		
		stringstream ss_filename_cmm;
		ss_filename_cmm << ss_filename.str() << "class_match_matrix.bin";
		util_write_binary(ss_filename_cmm.str(), (unsigned char *) spr->scrd->class_match_matrix, matrix_base_size * sizeof(bool));

		stringstream ss_filename_dm;
		ss_filename_dm << ss_filename.str() << "distance_matrix.bin";
		util_write_binary(ss_filename_dm.str(), (unsigned char*)spr->scrd->distance_matrix, matrix_base_size * sizeof(float));

		stringstream ss_filename_se;
		ss_filename_se << ss_filename.str() << "size_estimation_correction_dist.bin";
		util_write_binary(ss_filename_se.str(), (unsigned char*)spr->scrd->size_estimation_correction_dist_matrix, matrix_base_size * sizeof(float));

		spr->t_current = t_now;
		spr->t_c++;
	}
}

void statistic_position_regression_calculate(struct statistic_position_regression* spr) {
	size_t matrix_base_size = spr->camera_c * spr->scrd->cdh_max_size * spr->camera_c * spr->scrd->cdh_max_size;

	cudaMalloc(&spr->camera_positions_device, spr->camera_c * sizeof(struct vector3<float>));
	cudaMemcpyAsync(spr->camera_positions_device, spr->camera_positions, spr->camera_c * sizeof(struct vector3<float>), cudaMemcpyHostToDevice, cuda_streams[0]);

	cudaMalloc(&spr->camera_fov_factors_device, spr->camera_c * sizeof(float));
	cudaMemcpyAsync(spr->camera_fov_factors_device, spr->camera_fov_factors, spr->camera_c * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);

	cudaMalloc(&spr->camera_resolutions_x_device, spr->camera_c * sizeof(int));
	cudaMemcpyAsync(spr->camera_resolutions_x_device, spr->camera_resolutions_x, spr->camera_c * sizeof(int), cudaMemcpyHostToDevice, cuda_streams[0]);

	size_t parallel_cc_matrix_base_size = spr->camera_c * spr->camera_c * spr->parallel_c;

	spr->cc_matrix_avg_distance = (float *) malloc(parallel_cc_matrix_base_size * sizeof(float));
	cudaMalloc(&spr->cc_matrix_avg_distance_device, parallel_cc_matrix_base_size * sizeof(float));

	spr->cc_matrix_avg_correction_distance = (float *) malloc(parallel_cc_matrix_base_size * sizeof(float));
	cudaMalloc(&spr->cc_matrix_avg_correction_distance_device, parallel_cc_matrix_base_size * sizeof(float));

	cudaMalloc(&spr->camera_position_offsets_device, spr->parallel_c * spr->camera_c * sizeof(struct vector3<float>));
	

	struct vector3<size_t> ss_factor(2 * ceil(spr->parameter_search_space[0] / spr->stepsize) + 1, 2 * ceil(spr->parameter_search_space[1] / spr->stepsize) + 1, 2 * ceil(spr->parameter_search_space[2] / spr->stepsize) + 1);
	//struct vector3<size_t> ss_factor(2 * ceil(spr->parameter_search_space[0] / spr->stepsize), 2 * ceil(spr->parameter_search_space[1] / spr->stepsize), 2 * ceil(spr->parameter_search_space[2] / spr->stepsize));
	size_t search_space_size = ss_factor[0] * ss_factor[1] * ss_factor[2];
	size_t total_idx = search_space_size;
	for (int c = 1; c < spr->camera_c; c++) {
		total_idx *= search_space_size;
	}

	float min_norm_avg = FLT_MAX;
	size_t min_avg_p_idx = 0;

	float min_norm_avg_corr = FLT_MAX;
	size_t min_avg_corr_p_idx = 0;

	float min_norm_combined = FLT_MAX;
	size_t min_avg_combined_p_idx = 0;

	bool something_new = false;

	for (int i = 0; i < 10; i++) {

		for (size_t p_idx = 0; p_idx < total_idx; p_idx += spr->parallel_c) {
			logger(p_idx);

			cudaMemsetAsync(spr->cc_matrix_avg_distance_device, 0, parallel_cc_matrix_base_size * sizeof(float), cuda_streams[1]);
			cudaMemsetAsync(spr->cc_matrix_avg_correction_distance_device, 0, parallel_cc_matrix_base_size * sizeof(float), cuda_streams[2]);
			cudaStreamSynchronize(cuda_streams[1]);
			cudaStreamSynchronize(cuda_streams[2]);

			for (int t_c = 0; t_c < spr->t_samples_count; t_c++) {
				size_t out_len = 0;

				stringstream ss_filename;
				ss_filename << spr->temporary_storage_dir << t_c << "_";

				stringstream ss_filename_det_3d;
				ss_filename_det_3d << ss_filename.str() << "detections_3d.bin";
				util_read_binary(ss_filename_det_3d.str(), (unsigned char*)spr->scrd->detections_3d, &out_len);
				cudaMemcpyAsync(spr->scrd->detections_3d_device, spr->scrd->detections_3d, spr->camera_c * spr->scrd->cdh_max_size * sizeof(struct statistic_detection_matcher_3d_detection), cudaMemcpyHostToDevice, cuda_streams[0]);

				stringstream ss_filename_cmm;
				ss_filename_cmm << ss_filename.str() << "class_match_matrix.bin";
				util_read_binary(ss_filename.str(), (unsigned char*)spr->scrd->class_match_matrix, &out_len);
				cudaMemcpyAsync(spr->scrd->class_match_matrix_device, spr->scrd->class_match_matrix, matrix_base_size * sizeof(bool), cudaMemcpyHostToDevice, cuda_streams[0]);

				stringstream ss_filename_dm;
				ss_filename_dm << ss_filename.str() << "distance_matrix.bin";
				util_read_binary(ss_filename_dm.str(), (unsigned char*)spr->scrd->distance_matrix, &out_len);
				cudaMemcpyAsync(spr->scrd->distance_matrix_device, spr->scrd->distance_matrix, matrix_base_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);

				stringstream ss_filename_se;
				ss_filename_se << ss_filename.str() << "size_estimation_correction_dist.bin";
				util_read_binary(ss_filename_se.str(), (unsigned char*)spr->scrd->size_estimation_correction_dist_matrix, &out_len);
				cudaMemcpyAsync(spr->scrd->size_estimation_correction_dist_matrix_device, spr->scrd->size_estimation_correction_dist_matrix, matrix_base_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);

				cudaStreamSynchronize(cuda_streams[0]);
				
				statistics_position_regression_kernel_launch(spr->camera_positions_device, spr->camera_fov_factors_device, spr->camera_resolutions_x_device, spr->camera_c, spr->scrd->cdh_max_size, spr->scrd->detections_3d_device, spr->scrd->class_match_matrix_device, spr->scrd->distance_matrix_device, spr->scrd->size_estimation_correction_dist_matrix_device, spr->t_samples_count, spr->parallel_c, p_idx, total_idx, search_space_size, spr->parameter_search_space, spr->stepsize, struct vector2<size_t>(ss_factor[0], ss_factor[1]), spr->camera_position_offsets_device, spr->cc_matrix_avg_distance_device, spr->cc_matrix_avg_correction_distance_device);
			}

			cudaMemcpyAsync(spr->cc_matrix_avg_distance, spr->cc_matrix_avg_distance_device, parallel_cc_matrix_base_size * sizeof(float), cudaMemcpyDeviceToHost, cuda_streams[4]);
			cudaMemcpyAsync(spr->cc_matrix_avg_correction_distance, spr->cc_matrix_avg_correction_distance_device, parallel_cc_matrix_base_size * sizeof(float), cudaMemcpyDeviceToHost, cuda_streams[4]);
			cudaStreamSynchronize(cuda_streams[4]);

			for (int p = 0; p < spr->parallel_c; p++) {
				if (p_idx + p < total_idx) {
					float m_norm_avg_d = 0.0f;
					float m_norm_avg_corr_d = 0.0f;
					float m_norm_combined_d = 0.0f;
					for (int c = 0; c < spr->camera_c; c++) {
						float m_norm_l = 0.0f;
						float m_norm_c_l = 0.0f;

						for (int d = 0; d < spr->camera_c; d++) {
							float m_col = spr->cc_matrix_avg_distance[p * spr->camera_c * spr->camera_c + c * spr->camera_c + d];
							m_col *= m_col;
							m_norm_l += m_col;

							float m_col_c = spr->cc_matrix_avg_correction_distance[p * spr->camera_c * spr->camera_c + c * spr->camera_c + d];
							m_col_c *= m_col_c;
							m_norm_c_l += m_col_c;
						}

						m_norm_avg_d = max(sqrtf(m_norm_l), m_norm_avg_d);
						m_norm_avg_corr_d = max(sqrtf(m_norm_c_l), m_norm_avg_corr_d);
						m_norm_combined_d = max(max(m_norm_avg_corr_d, m_norm_avg_d), m_norm_combined_d);
					}
					if (m_norm_avg_d < min_norm_avg) {
						min_norm_avg = m_norm_avg_d;
						min_avg_p_idx = p_idx + p;
						logger("min_norm_avg", min_norm_avg);
						logger("idx", min_avg_p_idx);
					}
					if (m_norm_avg_corr_d < min_norm_avg_corr) {
						min_norm_avg_corr = m_norm_avg_corr_d;
						min_avg_corr_p_idx = p_idx + p;
						logger("min_norm_corr_avg", min_norm_avg_corr);
						logger("idx", min_avg_corr_p_idx);
					}
					if (m_norm_combined_d < min_norm_combined) {
						min_norm_combined = m_norm_combined_d;
						min_avg_combined_p_idx = p_idx + p;
						logger("min_norm_combined", min_norm_combined);
						logger("idx", min_avg_combined_p_idx);
						something_new = true;
					}
				}
			}
		}

		logger("min_norm_avg", min_norm_avg);
		logger("idx", min_avg_p_idx);

		logger("min_norm_corr_avg", min_norm_avg_corr);
		logger("idx", min_avg_corr_p_idx);

		logger("min_norm_combined", min_norm_combined);
		logger("idx", min_avg_combined_p_idx);

		if (something_new) {
			size_t cam_offset_base = min_avg_combined_p_idx;

			for (int c = 0; c < spr->camera_c; c++) {
				size_t c_idx = cam_offset_base % search_space_size;

				size_t idx_z = c_idx / (ss_factor[0] * ss_factor[1]);

				size_t idx__y = c_idx % ((ss_factor[0] * ss_factor[1]));
				size_t idx_y = idx__y / (ss_factor[0]);

				size_t idx_x = idx__y % ss_factor[0];

				spr->camera_positions[c] = spr->camera_positions[c] - -struct vector3<float>(-spr->parameter_search_space[0] + (float)idx_x * spr->stepsize, -spr->parameter_search_space[1] + (float)idx_y * spr->stepsize, -spr->parameter_search_space[2] + (float)idx_z * spr->stepsize);

				stringstream s_c;
				s_c << c << ": " << spr->camera_positions[c][0] << ", " << spr->camera_positions[c][1] << ", " << spr->camera_positions[c][2];

				logger(s_c.str());

				cam_offset_base /= search_space_size;
			}
			cudaMemcpyAsync(spr->camera_positions_device, spr->camera_positions, spr->camera_c * sizeof(struct vector3<float>), cudaMemcpyHostToDevice, cuda_streams[0]);
			something_new = false;
		} else {
			break;
		}
	}
}
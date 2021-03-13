#include "Statistic.h"

#include <math.h>
#include <cstdlib>

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

void statistic_angle_denoiser_update(struct statistic_angle_denoiser* sad, float angle) {
	sad->angle_distribution_idx_latest = (sad->angle_distribution_idx_latest + 1) % sad->angle_distribution_size;
	sad->angle_distribution[sad->angle_distribution_idx_latest] = angle;
	if (sad->angle_distribution[sad->angle_distribution_idx_latest] < 0) sad->angle_distribution[sad->angle_distribution_idx_latest] += 360.0f;
	float np_tmp = (float)sad->angle_distribution[sad->angle_distribution_idx_latest];
	sad->angle_stability = 1.0f;
	int tmp_d_c = sad->angle_distribution_idx_latest;
	for (int np_d = 0; np_d < sad->angle_distribution_size; np_d++) {
		float f_dt = 0.0f;
		float dt_s = 1 - (2 * (sad->angle_distribution[sad->angle_distribution_idx_latest] - sad->angle_distribution[tmp_d_c] > 0));
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
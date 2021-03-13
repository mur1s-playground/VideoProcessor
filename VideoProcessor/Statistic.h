#pragma once

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
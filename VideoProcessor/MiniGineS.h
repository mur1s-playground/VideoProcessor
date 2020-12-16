#pragma once

#include "Vector2.h"
#include "Vector3.h"

struct mini_gine_model_params {
	unsigned char r, g, b;
	float s;
};

struct mini_gine_model {
	unsigned int					id;

	struct vector2<unsigned int>	model_dimensions;
	unsigned int					model_rotations;
	unsigned int					model_animation_ticks;
	unsigned int					model_animation_stepsize;
	unsigned int					model_animation_type;
	unsigned int					model_params;

	unsigned int					model_positions;

	unsigned int					model_params_coi_offset_position;
};

struct mini_gine_entity {
	struct vector2<float>			position;
	float							scale;
	float							orientation;

	struct vector2<unsigned int>	crop_x;
	struct vector2<unsigned int>	crop_y;
	unsigned int					model_id;
	unsigned int					model_z;
	unsigned int					model_animation_offset;

	unsigned int					model_params_position;
	float							model_params_s_multiplier;
	float							model_params_s_falloff;
};
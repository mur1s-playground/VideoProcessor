#pragma once

#include "ApplicationGraph.h"
#include "VideoSource.h"

#include "Vector2.h"
#include "Vector3.h"
#include "BitField.h"
#include "Grid.h"

#include <map>
#include <string.h>

#include "MiniGineS.h"

enum mini_gine_entity_animation_types {
	MGEAT_MOVIE,
	MGEAT_JUST_ON,
	MGEAT_2COLOR_BLINK,
	MGEAT_SNAKE
};

struct mini_gine_entity_meta_animation {
	enum mini_gine_entity_animation_types animation_type;
	vector<float>						  animation_params;
};

struct mini_gine_entity_meta {
	vector<struct mini_gine_model_params>			mgmp;
	vector<struct mini_gine_entity_meta_animation>	mgema;
};

struct mini_gine {
	char*							config_path;

	map<string, unsigned int>		model2id;
	vector<struct mini_gine_model>	models;
	vector<struct mini_gine_entity>	entities;
	vector<struct mini_gine_entity_meta> entities_meta;

	unsigned int					models_position;
	unsigned int					entities_position;
	struct bit_field				bf_assets;
	struct bit_field				bf_rw;
	struct grid						gd;

	unsigned int					tick_counter;
	struct video_source* v_src_in;
	struct video_source* v_src_out;
};

void mini_gine_init(struct mini_gine* mg, const char *config_path);
void mini_gine_on_input_disconnect(struct application_graph_edge* edge);

void mini_gine_on_entity_update(struct mini_gine* mg);

DWORD* mini_gine_loop(LPVOID args);

void mini_gine_externalise(struct application_graph_node* agn, string& out_str);
void mini_gine_load(struct mini_gine* mg, ifstream& in_f);
void mini_gine_destroy(struct application_graph_node* agn);

/* MINI_GINE MODEL */
unsigned int mini_gine_model_add(struct mini_gine* mg, string model_name);
void mini_gine_model_remove(struct mini_gine* mg, const unsigned int model_id);

/* MINI_GINE ENTITY */
void mini_gine_entity_add(struct mini_gine* mg, string i_line);
void mini_gine_entity_remove(struct mini_gine* mg, const unsigned int entity_id);
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

struct mini_gine {
	char*							config_path;

	map<string, unsigned int>		model2id;
	vector<struct mini_gine_model>	models;
	vector<struct mini_gine_entity>	entities;

	unsigned int					models_position;
	unsigned int					entities_position;
	struct bit_field				bf_assets;
	struct bit_field				bf_rw;
	struct grid						gd;

	struct video_source* v_src_out;
};

void mini_gine_init(struct mini_gine* mg, const char *config_path);
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
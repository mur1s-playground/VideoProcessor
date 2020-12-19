#include "MiniGine.h"

#include <stdio.h>
#include <sstream>
#include <fstream>
#include "MiniGineKernel.h"
#include "Logger.h"

#include <opencv2/imgcodecs.hpp>

#include "MainUI.h"

void mini_gine_init(struct mini_gine* mg, const char *config_path) {
	if (config_path != nullptr) {
		mg->config_path = new char[strlen(config_path) + 1];
		snprintf(mg->config_path, strlen(config_path) + 1, "%s", config_path);

		mg->tick_counter = 0;
		
		stringstream s_fullpath;
		s_fullpath << mg->config_path;

		std::ifstream infile;
		infile.open(s_fullpath.str().c_str(), std::ios_base::in);
		string i_line;

		mg->models_position = 0;
		mg->entities_position = 0;

		mg->v_src_in = nullptr;

		bit_field_init(&mg->bf_assets, 16, 1024);
		bit_field_register_device(&mg->bf_assets, 0);

		bit_field_init(&mg->bf_rw, 16, 1024);
		bit_field_register_device(&mg->bf_rw, 0);
		
		while (std::getline(infile, i_line)) {
			if (strlen(i_line.c_str()) == 0) return;
			
			mini_gine_entity_add(mg, i_line);
		}
	} else {
		mg->config_path = nullptr;
	}
	
}

void mini_gine_on_input_disconnect(struct application_graph_edge* edge) {
	struct mini_gine* mg = (struct mini_gine*)edge->to.first->component;
	struct video_source* vs = (struct video_source*)edge->from.first->component;
	if (vs == mg->v_src_in) {
		mg->v_src_in = nullptr;
		mg->tick_counter = 0;
	}
}

void mini_gine_on_entity_update(struct mini_gine* mg) {
	unsigned int entities_size_in_mem = mg->entities.size() * sizeof(struct mini_gine_entity);
	unsigned int entities_size_in_bf = (unsigned int)ceilf((float)entities_size_in_mem / (float)sizeof(unsigned int));
	bit_field_update_bulk(&mg->bf_rw, mg->entities_position, (unsigned int*)mg->entities.data(), entities_size_in_bf, entities_size_in_mem);
}

DWORD* mini_gine_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct mini_gine* mg = (struct mini_gine*)agn->component;

	unsigned int models_size_in_mem = mg->models.size() * sizeof(struct mini_gine_model);
	unsigned int models_size_in_bf = (unsigned int)ceilf((float)models_size_in_mem / (float)sizeof(unsigned int));
	mg->models_position = bit_field_add_bulk(&mg->bf_assets, (unsigned int *)mg->models.data(), models_size_in_bf, models_size_in_mem) + 1;

	bit_field_update_device(&mg->bf_assets, 0);

	unsigned int entities_size_in_mem = mg->entities.size() * sizeof(struct mini_gine_entity);
	unsigned int entities_size_in_bf = (unsigned int)ceilf((float)entities_size_in_mem / (float)sizeof(unsigned int));
	mg->entities_position = bit_field_add_bulk(&mg->bf_rw, (unsigned int*)mg->entities.data(), entities_size_in_bf, entities_size_in_mem) + 1;

	grid_init(&mg->bf_rw, &mg->gd, struct vector3<float>((float)mg->v_src_out->video_width, (float)mg->v_src_out->video_height, 1.0f), struct vector3<float>(64.0f, 64.0f, 1.0f), struct vector3<float>(0, 0, 0));

	int z_index = 0;
	while (z_index < 256) {
		for (int e = 0; e < mg->entities.size(); e++) {
			struct mini_gine_entity* entity = &mg->entities[e];
			//unsigned int grid_index = grid_get_index(mg->bf_rw.data, mg->gd.position_in_bf, { entity->position[0], entity->position[1], 0.0f });
			if (entity->model_z == z_index) {
				grid_object_add(&mg->bf_rw, mg->bf_rw.data, mg->gd.position_in_bf, { entity->position[0], entity->position[1], 0.0f }, { entity->scale , entity->scale, 1.0f }, { 0.0f, 0.0f, 0.0f }, { (float)(entity->crop_x[1] - entity->crop_x[0]), (float)(entity->crop_y[1] - entity->crop_y[0]), 0.0f }, e);
			}
		}
		z_index++;
	}

	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		struct mini_gine_model* models = (struct mini_gine_model*)&mg->bf_assets.data[mg->models_position];
		struct mini_gine_entity* entity = mg->entities.data();// (struct mini_gine_entity*)&mg->bf_rw.data[mg->entities_position];

		int next_id = 0;
		if (mg->v_src_in != nullptr) {
			shared_memory_buffer_try_r(mg->v_src_in->smb, mg->v_src_in->smb_framecount, true, 8);
			next_id = mg->v_src_in->smb->p_buf_c[mg->v_src_in->smb_framecount * mg->v_src_in->video_channels * mg->v_src_in->video_height * mg->v_src_in->video_width + ((mg->v_src_in->smb_framecount + 1) * 2)];
			shared_memory_buffer_release_r(mg->v_src_in->smb, mg->v_src_in->smb_framecount);
			shared_memory_buffer_try_r(mg->v_src_in->smb, next_id, true, 8);
		}

		for (int e = 0; e < mg->entities.size(); e++) {
			if (models[entity->model_id].model_params > 0) {
				struct mini_gine_model_params* mgmp = (struct mini_gine_model_params*)&mg->bf_rw.data[entity->model_params_position];
				for (int p = 0; p < models[entity->model_id].model_params; p++) {
					struct mini_gine_entity_meta* mgem = &mg->entities_meta[e];
					if (mgem->mgema[p].animation_type == MGEAT_MOVIE) {
						//Movie
						if (mg->v_src_in != nullptr) {
							unsigned int* coio = &mg->bf_assets.data[models[entity->model_id].model_params_coi_offset_position];
							int coio_idx = (((int)entity->orientation / 10 / (36 / models[entity->model_id].model_rotations)) % models[entity->model_id].model_rotations) * models[entity->model_id].model_animation_ticks * (models[entity->model_id].model_params + 2);
							int x = entity->position[0] + coio[coio_idx + p * 2 + 1] * entity->scale;
							int y = entity->position[1] + coio[coio_idx + p * 2] * entity->scale;
							if (x >= 0 && x < mg->v_src_in->video_width && y >= 0 && y < mg->v_src_in->video_height) {
								mgmp->b = mg->v_src_in->smb->p_buf_c[next_id * mg->v_src_in->video_channels * mg->v_src_in->video_height * mg->v_src_in->video_width + y * mg->v_src_in->video_channels * mg->v_src_in->video_width + x * mg->v_src_in->video_channels];
								mgmp->g = mg->v_src_in->smb->p_buf_c[next_id * mg->v_src_in->video_channels * mg->v_src_in->video_height * mg->v_src_in->video_width + y * mg->v_src_in->video_channels * mg->v_src_in->video_width + x * mg->v_src_in->video_channels + 1];
								mgmp->r = mg->v_src_in->smb->p_buf_c[next_id * mg->v_src_in->video_channels * mg->v_src_in->video_height * mg->v_src_in->video_width + y * mg->v_src_in->video_channels * mg->v_src_in->video_width + x * mg->v_src_in->video_channels + 2];
								mgmp->s = 1.0f;
							}
						}
					} else if (mgem->mgema[p].animation_type == MGEAT_JUST_ON) {
						//Just on
						mgmp->r = mgem->mgmp[p].r;
						mgmp->g = mgem->mgmp[p].g;
						mgmp->b = mgem->mgmp[p].b;
						mgmp->s = mgem->mgmp[p].s;
						/*
					} else if (mgem->mgema[p].animation_type == 2) {
						//Blinking -> 2-Color blink: [0]:[0], [1]:[1], [2]:[2], [3]:1.0f, [4]:[0], [5]:0.0f, [6]:0.0f
						mgmp->r = mgem->mgmp[p].r;
						mgmp->g = mgem->mgmp[p].g;
						mgmp->b = mgem->mgmp[p].b;
						if ((mg->tick_counter + (int)mgem->mgema[p].animation_params[0]) % (int)mgem->mgema[p].animation_params[1] <= (int)mgem->mgema[p].animation_params[2]) {
							mgmp->s = mgem->mgmp[p].s;
						} else {
							mgmp->s = 0.0f;
						}
						*/
					} else if (mgem->mgema[p].animation_type == MGEAT_2COLOR_BLINK) {
						//2-Color Blink
						if ((mg->tick_counter + (int)mgem->mgema[p].animation_params[4]) % (int)mgem->mgema[p].animation_params[5] < (int)mgem->mgema[p].animation_params[6]) {
							mgmp->r = (unsigned char)mgem->mgema[p].animation_params[7];
							mgmp->g = (unsigned char)mgem->mgema[p].animation_params[8];
							mgmp->b = (unsigned char)mgem->mgema[p].animation_params[9];
						} else {
							mgmp->r = mgem->mgmp[p].r;
							mgmp->g = mgem->mgmp[p].g;
							mgmp->b = mgem->mgmp[p].b;
						}
						if ((mg->tick_counter + (int)mgem->mgema[p].animation_params[0]) % (int)mgem->mgema[p].animation_params[1] <= (int)mgem->mgema[p].animation_params[2]) {
							mgmp->s = (rand() / (float)RAND_MAX <= mgem->mgema[p].animation_params[3]) * (!mgmp->s) * mgem->mgmp[p].s;
						}
					} else if (mgem->mgema[p].animation_type == MGEAT_SNAKE) {
						//Snake
						mgmp->r = mgem->mgmp[p].r;
						mgmp->g = mgem->mgmp[p].g;
						mgmp->b = mgem->mgmp[p].b;
						if ((mg->tick_counter + (int)mgem->mgema[p].animation_params[0]) % (int)mgem->mgema[p].animation_params[1] == p/(int)mgem->mgema[p].animation_params[2]) {
							mgmp->s = (!mgmp->s) * mgem->mgmp[p].s;
						}
					}
						
					mgmp++;
				}

				/*
				if (mg->entities_metaentity->model_params_animation_type == 0) {
					if (mg->v_src_in != nullptr) {
						unsigned int* coio = &mg->bf_assets.data[models[entity->model_id].model_params_coi_offset_position];
						for (int p = 0; p < models[entity->model_id].model_params; p++) {
							int coio_idx = (((int)entity->orientation / 10 / (36 / models[entity->model_id].model_rotations)) % models[entity->model_id].model_rotations) * models[entity->model_id].model_animation_ticks * (models[entity->model_id].model_params + 2);
							int x = entity->position[0] + coio[coio_idx + p * 2 + 1] * entity->scale;
							int y = entity->position[1] + coio[coio_idx + p * 2] * entity->scale;
							if (x >= 0 && x < mg->v_src_in->video_width && y >= 0 && y < mg->v_src_in->video_height) {
								mgmp->b = mg->v_src_in->smb->p_buf_c[next_id * mg->v_src_in->video_channels * mg->v_src_in->video_height * mg->v_src_in->video_width + y * mg->v_src_in->video_channels * mg->v_src_in->video_width + x * mg->v_src_in->video_channels];
								mgmp->g = mg->v_src_in->smb->p_buf_c[next_id * mg->v_src_in->video_channels * mg->v_src_in->video_height * mg->v_src_in->video_width + y * mg->v_src_in->video_channels * mg->v_src_in->video_width + x * mg->v_src_in->video_channels + 1];
								mgmp->r = mg->v_src_in->smb->p_buf_c[next_id * mg->v_src_in->video_channels * mg->v_src_in->video_height * mg->v_src_in->video_width + y * mg->v_src_in->video_channels * mg->v_src_in->video_width + x * mg->v_src_in->video_channels + 2];
								mgmp->s = 1.0f;
							}
							else {
								mgmp->s = 0.0f;
							}
							mgmp++;
						}
					}
				} else if (entity->model_params_animation_type == 1) {
					//Just all on
					for (int p = 0; p < models[entity->model_id].model_params; p++) {
						mgmp->r = entity->model_params_animation_type_params[0];
						mgmp->g = entity->model_params_animation_type_params[1];
						mgmp->b = entity->model_params_animation_type_params[2];
						mgmp->s = 1.0f;
						mgmp++;
					}
				} else if (entity->model_params_animation_type == 2) {
					//Blinking
					for (int p = 0; p < models[entity->model_id].model_params; p++) {
						mgmp->r = entity->model_params_animation_type_params[0];
						mgmp->g = entity->model_params_animation_type_params[1];
						mgmp->b = entity->model_params_animation_type_params[2];
						if ((mg->tick_counter + (int)entity->model_params_animation_type_params_2[0]) % (int)entity->model_params_animation_type_params_2[1] < (int)entity->model_params_animation_type_params_2[2]) {
							mgmp->s = 1.0f;
						} else {
							mgmp->s = 0.0f;
						}
						mgmp++;
					}
				} else if (entity->model_params_animation_type == 3) {
					//Color-White Blink
					for (int p = 0; p < models[entity->model_id].model_params; p++) {
						if (mg->tick_counter % (int)entity->model_params_animation_type_params_2[2] < (int)entity->model_params_animation_type_params_2[2]/2) {
							mgmp->r = 255;
							mgmp->g = 255;
							mgmp->b = 255;
						} else {
							mgmp->r = entity->model_params_animation_type_params[0];
							mgmp->g = entity->model_params_animation_type_params[1];
							mgmp->b = entity->model_params_animation_type_params[2];
						}
						if ((mg->tick_counter + (int)entity->model_params_animation_type_params_2[0]) % (int)entity->model_params_animation_type_params_2[1] == 0) {
							mgmp->s = (rand() / (float)RAND_MAX < 0.5) * (!mgmp->s);
						}
						mgmp++;
					}	
				} else if (entity->model_params_animation_type == 4) {
					//snake
					for (int p = 0; p < models[entity->model_id].model_params; p++) {
						mgmp->r = entity->model_params_animation_type_params[0];
						mgmp->g = entity->model_params_animation_type_params[1];
						mgmp->b = entity->model_params_animation_type_params[2];
						if ((mg->tick_counter + (int)entity->model_params_animation_type_params_2[0]) % (int)entity->model_params_animation_type_params_2[1] == p / (int)entity->model_params_animation_type_params_2[2]) {
							mgmp->s = !mgmp->s;
						}
						mgmp++;
					}
				}
				*/
				/*
					//Left Outside
					if (p == 0 || p == 7 || p == 15 || p == 21 || p == 27 || p == 33 || p == 37 || p == 41 || p == 44 || p == 47 || p == 49
									//Top
									|| p == 51 ||
									//Right Outside
									p == 50 || p == 48 || p == 46 || p == 43 || p == 40 || p == 36 || p == 32 || p == 26 || p == 20 || p == 13 || p == 6
									) {
									mgmp->r = 100;
									mgmp->g = 90;
									mgmp->b = 0;
									mgmp->s = 1.0f;
								} else {
									mgmp->r = 100;
									mgmp->g = 100;
									mgmp->b = 150;
									mgmp->s = 1.0f;
									
									if (mg->tick_counter % 240 < 120) {
										mgmp->s = (mg->tick_counter % 240) / 120.0f;
									}
									else {
										mgmp->s = 1.0 - ((mg->tick_counter % 240) / 120.0f);
									}
									
								}
							} else if (entity->model_id == 6) {
								mgmp->r = 255;
								mgmp->g = 0;
								mgmp->b = 0;
								entity->model_params_s_multiplier = 15.0f;
								entity->model_params_s_falloff = 1.0f;
								if (mg->tick_counter == 0) {
									mgmp->s = 0;
								}
								if ((mg->tick_counter + e) % 10 == p / 6) {
									mgmp->s = !mgmp->s;
								}
							}
						}
				*/
				unsigned int params_size = (unsigned int)ceilf((float)((sizeof(struct mini_gine_model_params)) * models[entity->model_id].model_params) / (float)sizeof(unsigned int));
				
				bit_field_invalidate_bulk(&mg->bf_rw, entity->model_params_position, params_size);
			}
			entity++;
		}

		if (mg->v_src_in != nullptr) {
			shared_memory_buffer_release_r(mg->v_src_in->smb, next_id);
		}

		bit_field_update_device(&mg->bf_rw, 0);

		gpu_memory_buffer_try_r(mg->v_src_out->gmb, mg->v_src_out->gmb->slots, true, 8);
		int next_gpu_out_id = (mg->v_src_out->gmb->p_rw[2 * (mg->v_src_out->gmb->slots + 1)] + 1) % mg->v_src_out->gmb->slots;
		gpu_memory_buffer_release_r(mg->v_src_out->gmb, mg->v_src_out->gmb->slots);

		gpu_memory_buffer_try_rw(mg->v_src_out->gmb, next_gpu_out_id, true, 8);

		mini_gine_draw_entities_kernel_launch(mg->bf_assets.device_data[0], mg->models_position,
			mg->bf_rw.device_data[0], mg->entities_position, mg->gd.position_in_bf, mg->gd.data_position_in_bf,
			&mg->v_src_out->gmb->p_device[next_gpu_out_id * mg->v_src_out->video_channels * mg->v_src_out->video_width * mg->v_src_out->video_height], mg->v_src_out->video_width, mg->v_src_out->video_height, mg->v_src_out->video_channels,
			mg->tick_counter);

		gpu_memory_buffer_set_time(mg->v_src_out->gmb, next_gpu_out_id, application_graph_tps_balancer_get_time());
		gpu_memory_buffer_release_rw(mg->v_src_out->gmb, next_gpu_out_id);

		gpu_memory_buffer_try_rw(mg->v_src_out->gmb, mg->v_src_out->gmb->slots, true, 8);
		mg->v_src_out->gmb->p_rw[2 * (mg->v_src_out->gmb->slots + 1)] = next_gpu_out_id;
		gpu_memory_buffer_release_rw(mg->v_src_out->gmb, mg->v_src_out->gmb->slots);

		mg->tick_counter++;
		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}

	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void mini_gine_externalise(struct application_graph_node* agn, string& out_str) {
	struct mini_gine* mg = (struct mini_gine*)agn->component;

	stringstream s_out;
	s_out << mg->config_path << std::endl;
	
	out_str = s_out.str();
}

void mini_gine_load(struct mini_gine* mg, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	mini_gine_init(mg, line.c_str());
}

void mini_gine_destroy(struct application_graph_node* agn) {

}

/*------------------------------------------------------------------------------------*/
/*---------------------------------- HELPER ------------------------------------------*/
/*------------------------------------------------------------------------------------*/
std::string& ltrim(std::string& str, const std::string& chars) {
	str.erase(0, str.find_first_not_of(chars));
	return str;
}

std::string& rtrim(std::string& str, const std::string& chars) {
	str.erase(str.find_last_not_of(chars) + 1);
	return str;
}

std::string& trim(std::string& str, const std::string& chars) {
	return ltrim(rtrim(str, chars), chars);
}

vector<pair<string, string>> get_cfg_key_value_pairs(string filepath) {
	vector<pair<string, string>> result;

	ifstream file(filepath);
	string filecontent;
	string chars(" \n\r\t");
	if (file.is_open()) {
		while (std::getline(file, filecontent)) {
			if (filecontent.size() > 0) {
				size_t last_pos = 0;
				size_t pos = filecontent.find(':');
				if (pos != string::npos) {
					string name = filecontent.substr(0, pos);
					name = trim(name, chars);
					string value = filecontent.substr(pos + 1);
					value = trim(value, chars);
					result.push_back(pair<string, string>(name, value));
					printf("cfg key: %s, value: %s\n", name.c_str(), value.c_str());
				}
			}
		}
	}
	return result;
}
/*------------------------------------------------------------------------------------*/


unsigned int mini_gine_model_add(struct mini_gine* mg, string model_name) {
	map<string, unsigned int>::iterator model_it = mg->model2id.find(model_name);
	if (model_it != mg->model2id.end()) {
		return model_it->second;
	} else {
		mg->model2id[model_name] = mg->model2id.size()-1;

		string cfg_path(mg->config_path);
		int cfg_dir = cfg_path.find_last_of("\\");

		stringstream model_dir;
		model_dir << cfg_path.substr(0, cfg_dir) << "\\models\\" << model_name << "/";

		stringstream model_cfg;
		model_cfg << cfg_path.substr(0, cfg_dir) << "\\models\\" << model_name << ".cfg";

		vector<pair<string, string>> cfg_key_value = get_cfg_key_value_pairs(model_cfg.str());

		struct mini_gine_model mgm;

		string chars(" \n\r\t");
		for (int i = 0; i < cfg_key_value.size(); i++) {
			string key = cfg_key_value[i].first;
			string value = cfg_key_value[i].second;
			
			if (key.compare("model_dimensions") == 0) {
				string first, second;
				int comma = value.find_first_of(",");
				first = value.substr(0, comma);
				first = trim(first, chars);
				second = value.substr(comma + 1);
				second = trim(second, chars);
				int w = stoi(first);
				int h = stoi(second);
				mgm.model_dimensions = vector2<unsigned int>(w, h);
			} else if (key.compare("model_rotations") == 0) {
				mgm.model_rotations = stoi(value);
			} else if (key.compare("model_animation_ticks") == 0) {
				mgm.model_animation_ticks = stoi(value);
			} else if (key.compare("model_animation_stepsize") == 0) {
				mgm.model_animation_stepsize = stoi(value);
			} else if (key.compare("model_params") == 0) {
				mgm.model_params = stoi(value);
			} else if (key.compare("file_prefix") == 0) {
				string first;
				int hash = value.find_first_of("#");
				if (hash > 0) {
					first = value.substr(0, hash);
				} else {
					first = "";
				}
				
				int count = mgm.model_rotations * mgm.model_animation_ticks;
				if (mgm.model_params > 0) {
					count *= (mgm.model_params + 2);
					mgm.model_params_coi_offset_position = bit_field_add_bulk_zero(&mg->bf_assets, mgm.model_rotations * mgm.model_animation_ticks * mgm.model_params * 2) + 1;
				} else {
					mgm.model_params_coi_offset_position = 0;
				}
				
				mgm.model_positions = bit_field_add_bulk_zero(&mg->bf_assets, count) + 1;
				
				unsigned int img_in_bf = (unsigned int)ceilf((float)(mgm.model_dimensions[0] * mgm.model_dimensions[1] * 4 * sizeof(unsigned char)) / (float)sizeof(unsigned int));

				for (int k = 0; k < count; k++) {
					int leading_zeros = 4;
					stringstream n;
					n << k;

					stringstream sr;
					sr << cfg_path.substr(0, cfg_dir) << "\\models\\" << model_name << "\\" << first;
					for (int j = strlen(n.str().c_str()); j < leading_zeros; j++) {
						sr << 0;
					}
					sr << k << ".png";
					Mat img = cv::imread(sr.str(), IMREAD_UNCHANGED);
					if (img.data == NULL) logger("error loading image");
					unsigned int img_pos_in_bf = bit_field_add_bulk(&mg->bf_assets, (unsigned int *) img.data, img_in_bf, mgm.model_dimensions[0] * mgm.model_dimensions[1] * 4 * sizeof(unsigned char)) + 1;

					unsigned int* m_ap = &mg->bf_assets.data[mgm.model_positions];
					m_ap[k] = img_pos_in_bf;
				}

				if (mgm.model_params > 0) {
					unsigned int* coio = &mg->bf_assets.data[mgm.model_params_coi_offset_position];
					unsigned int* m_ap = &mg->bf_assets.data[mgm.model_positions];
					for (int r = 0; r < mgm.model_rotations; r++) {
						for (int a = 0; a < mgm.model_animation_ticks; a++) {
							for (int p = 0; p < mgm.model_params; p++) {
								unsigned char* zero = (unsigned char*)&mg->bf_assets.data[m_ap[r * mgm.model_animation_ticks * (mgm.model_params + 2) + a * (mgm.model_params + 2)]];
								unsigned char* dim = (unsigned char*)&mg->bf_assets.data[m_ap[r * mgm.model_animation_ticks * (mgm.model_params + 2) + a * (mgm.model_params + 2) + 1 + p]];
								int max_value_min_x = -255;
								int min_x = mgm.model_dimensions[0];
								int max_value_max_x = -255;
								int max_x = 0;
								int max_value_min_y = -255;
								int min_y = mgm.model_dimensions[1];
								int max_value_max_y = -255;
								int max_y = 0;
								int thres = 10;
								//int max_value = -255;
								for (int row = 0; row < mgm.model_dimensions[1]; row++) {
									for (int col = 0; col < mgm.model_dimensions[0]; col++) {
										int cur_base_idx = row * mgm.model_dimensions[0] * 4 + col * 4;
										for (int ch = 0; ch < 3; ch++) {
											int diff = (int)dim[cur_base_idx + ch] - (int)zero[cur_base_idx + ch];
											if (diff > thres && col < min_x) {
												max_value_min_x = diff;
												min_x = col;
											}
											if (diff > thres && col > max_x) {
												max_value_max_x = diff;
												max_x = col;
											}
											if (diff > thres && row < min_y) {
												max_value_min_y = diff;
												min_y = row;
											}
											if (diff > thres && row > max_y) {
												max_value_max_y = diff;
												max_y = row;
											}
											/*
											if ((int)dim[cur_base_idx + ch] - (int)zero[cur_base_idx + ch] > max_value) {
												max_value = (int)dim[cur_base_idx + ch] - (int)zero[cur_base_idx + ch];
												coio[r * mgm.model_animation_ticks * mgm.model_params * 2 + a * mgm.model_params * 2 + p * 2] = row;
												coio[r * mgm.model_animation_ticks * mgm.model_params * 2 + a * mgm.model_params * 2 + p * 2 + 1] = col;
											}
											*/
										}
									}
								}
								coio[r * mgm.model_animation_ticks * mgm.model_params * 2 + a * mgm.model_params * 2 + p * 2] = min_y + 0.5*(max_y - min_y);
								coio[r * mgm.model_animation_ticks * mgm.model_params * 2 + a * mgm.model_params * 2 + p * 2 + 1] = min_x + 0.5*(max_x - min_x);
							}
						}
					}
				}
			}
		}
		mg->models.push_back(mgm);
		return mg->model2id[model_name];
	}
}

void mini_gine_model_remove(struct mini_gine* mg, const unsigned int model_id) {

}

/* MINI_GINE ENTITY */
void mini_gine_entity_add(struct mini_gine* mg, string i_line) {
	struct mini_gine_entity mge;

	int start = 0;
	int end = i_line.find_first_of(",", start);
	string part;
	int idx = 0;

	while (end != string::npos) {
		part = i_line.substr(start, end - start);
		start = end + 1;
		end = i_line.find_first_of(",", start);
		switch (idx) {
		case 0:
			mge.model_id = mini_gine_model_add(mg, part);
			break;
		case 1:
			mge.position[0] = stof(part);
			break;
		case 2:
			mge.position[1] = stof(part);
			break;
		case 3:
			mge.scale = stof(part);
			break;
		case 4:
			mge.orientation = stof(part);
			break;
		case 5:
			mge.model_z = stoi(part);
			break;
		case 6:
			mge.model_animation_offset = stoi(part);
			break;
		case 7:
			mge.crop_x[0] = stoi(part);
			break;
		case 8:
			mge.crop_x[1] = stoi(part);
			break;
		case 9:
			mge.crop_y[0] = stoi(part);
			break;
		}
		idx++;
	}
	part = i_line.substr(start, end);
	mge.crop_y[1] = stoi(part);
	if (mg->models[mge.model_id].model_params > 0) {
		unsigned int params_size = (unsigned int)ceilf((float)((sizeof(struct mini_gine_model_params)) * mg->models[mge.model_id].model_params) / (float)sizeof(unsigned int));
		mge.model_params_position = bit_field_add_bulk_zero(&mg->bf_rw, params_size) + 1;
	} else {
		mge.model_params_position = 0;
		
	}

	struct mini_gine_entity_meta mgem;
	for (int i = 0; i < mg->models[mge.model_id].model_params; i++) {
		struct mini_gine_model_params mgmp;
		mgmp.b = 255;
		mgmp.g = 255;
		mgmp.r = 255;
		mgmp.s = 0;
		mgem.mgmp.push_back(mgmp);
		for (int p = 0; p < mg->models[mge.model_id].model_params; p++) {
			struct mini_gine_entity_meta_animation mgema;
			mgema.animation_type = MGEAT_JUST_ON;
			mgem.mgema.push_back(mgema);
		}
	}
	mg->entities_meta.push_back(mgem);

	mge.model_params_s_multiplier = 1.0f;
	mge.model_params_s_falloff = 0.0f;
	
	mg->entities.push_back(mge);
}

void mini_gine_entity_remove(struct mini_gine* mg, const unsigned int entity_id) {

}
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
		
		stringstream s_fullpath;
		s_fullpath << mg->config_path;

		std::ifstream infile;
		infile.open(s_fullpath.str().c_str(), std::ios_base::in);
		string i_line;

		mg->models_position = 0;
		mg->entities_position = 0;

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

	unsigned int tick_counter = 0;

	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

		struct mini_gine_model* models = (struct mini_gine_model*)&mg->bf_assets.data[mg->models_position];
		struct mini_gine_entity* entity = (struct mini_gine_entity *)&mg->bf_rw.data[mg->entities_position];
		for (int e = 0; e < mg->entities.size(); e++) {
			if (models[entity->model_id].model_params > 0) {
				struct mini_gine_model_params* mgmp = (struct mini_gine_model_params*)&mg->bf_rw.data[entity->model_params_position];
				for (int p = 0; p < models[entity->model_id].model_params; p++) {
					if (models[entity->model_id].model_params == 14) {
						if (tick_counter == 0) {
							mgmp->r = 255.0 * (e % 3 == 0);
							mgmp->g = 255.0 * (e % 3 == 1);
							mgmp->b = 255.0 * (e % 3 == 2);
							//mgmp->r = (rand() / (float)RAND_MAX) * 255.0f;
							//mgmp->g = (rand() / (float)RAND_MAX) * 255.0f;
							//mgmp->b = (rand() / (float)RAND_MAX) * 255.0f;
							//if (p % 2 == 0) mgmp->s = 1.0 * (e % 2 == 0);
							//if (p % 2 == 1) mgmp->s = 1.0 * (e % 2 == 1);

							/*
							if (p < 3) mgmp->s = 1.0 * (e % 2 == 0);
							if (p >= 3 && p < 6) mgmp->s = 1.0 * (e % 2 == 1);
							if (p >= 6 && p < 8) mgmp->s = 1.0 *(e % 2 == 0);
							if (p >= 8 && p < 10) mgmp->s = 1.0 * (e % 2 == 1);
							if (p >= 10) mgmp->s = 1.0 * (e % 2 == 0);
							*/
						}
						/*
						if (tick_counter % 10 == 0) {
							mgmp->s = !mgmp->s;
						}*/
						
						if (tick_counter % (models[entity->model_id].model_params * 10) == (p * 10 + e * 10) % (models[entity->model_id].model_params * 10)) {
							mgmp->s = 1.0;
							if (tick_counter > 0) {
								if (p == 0) {
									mgmp[models[entity->model_id].model_params - 1].s = 0.0f;
								} else {
									mgmp--;
									mgmp->s = 0.0f;
									mgmp++;
								}
							}
						}
						
					} else {
						if (tick_counter == 0) {
							mgmp->r = 0;
							mgmp->g = 0;
							mgmp->b = 255.0f;
							mgmp->s = 1.0f;
						}
						if (tick_counter % 8 == 0) {
							mgmp->r = (mgmp->r == 0) * 255.0f;
							mgmp->g = (mgmp->g == 0) * 255.0f;
						}
						if (tick_counter % 4 == 0) {
							mgmp->s = (rand() / (float)RAND_MAX < 0.5) * (!mgmp->s);
						}
					}
					/*
					if (tick_counter % 30 == (0 + e * 10) % 30) {
						if (p <= 3) {
							mgmp->s = 0.5;
						} else {
							mgmp->s = 0.0;
						}
					} else if (tick_counter % 30 == (10 + e * 10) % 30) {
						if (p <= 3) {
							mgmp->s = 0.0;
						} else if (p <= 6) {
							mgmp->s = 0.5;
						} else {
							mgmp->s = 0.0;
						}
					} else if (tick_counter % 30 == (20 + e * 10) % 30){
						if (p <= 6) {
							mgmp->s = 0.0;
						} else {
							mgmp->s = 0.25;
						}
					}
					*/
					
					mgmp++;
				}
				unsigned int params_size = (unsigned int)ceilf((float)((sizeof(struct mini_gine_model_params)) * models[entity->model_id].model_params) / (float)sizeof(unsigned int));
				
				bit_field_invalidate_bulk(&mg->bf_rw, entity->model_params_position, params_size);
			}
			entity++;
		}

		bit_field_update_device(&mg->bf_rw, 0);

		gpu_memory_buffer_try_r(mg->v_src_out->gmb, mg->v_src_out->gmb->slots, true, 8);
		int next_gpu_out_id = (mg->v_src_out->gmb->p_rw[2 * (mg->v_src_out->gmb->slots + 1)] + 1) % mg->v_src_out->gmb->slots;
		gpu_memory_buffer_release_r(mg->v_src_out->gmb, mg->v_src_out->gmb->slots);

		gpu_memory_buffer_try_rw(mg->v_src_out->gmb, next_gpu_out_id, true, 8);

		mini_gine_draw_entities_kernel_launch(mg->bf_assets.device_data[0], mg->models_position,
			mg->bf_rw.device_data[0], mg->entities_position, mg->entities.size(),
			&mg->v_src_out->gmb->p_device[next_gpu_out_id * mg->v_src_out->video_channels * mg->v_src_out->video_width * mg->v_src_out->video_height], mg->v_src_out->video_width, mg->v_src_out->video_height, mg->v_src_out->video_channels,
			tick_counter);

		gpu_memory_buffer_set_time(mg->v_src_out->gmb, next_gpu_out_id, application_graph_tps_balancer_get_time());
		gpu_memory_buffer_release_rw(mg->v_src_out->gmb, next_gpu_out_id);

		gpu_memory_buffer_try_rw(mg->v_src_out->gmb, mg->v_src_out->gmb->slots, true, 8);
		mg->v_src_out->gmb->p_rw[2 * (mg->v_src_out->gmb->slots + 1)] = next_gpu_out_id;
		gpu_memory_buffer_release_rw(mg->v_src_out->gmb, mg->v_src_out->gmb->slots);

		tick_counter++;
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
	mg->entities.push_back(mge);
}

void mini_gine_entity_remove(struct mini_gine* mg, const unsigned int entity_id) {

}
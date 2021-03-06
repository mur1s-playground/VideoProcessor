#include "AudioVisualKernel.h"

#include "GPUAudioVisual.h"

#include "CUDAStreamHandler.h"
#include "cuda_runtime.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "MainUI.h"
#include "ApplicationGraph.h"

#include "Windows.h"

#include <sstream>
#include <fstream>

#include "Logger.h"

void gpu_audiovisual_init(struct gpu_audiovisual* gav, const char *name, int dft_size) {
	gav->name = string(name);
	gav->dft_size = dft_size;

	gav->mats_in = new Mat[gav->frame_names.size()];
	for (int i = 0; i < gav->frame_names.size(); i++) {
		gav->mats_in[i] = cv::imread(gav->frame_names[i], IMREAD_COLOR);
	}

	gav->theme_count = gav->frame_names.size() / 9;

	struct gpu_memory_buffer* dft_out = new gpu_memory_buffer();

	gpu_memory_buffer_init(dft_out, name, gav->dft_size * sizeof(float), 1, sizeof(int));

	gav->dft.d_grid = 628;
	
	cudaMalloc((void**)&gav->dft.sinf_d, gav->dft.d_grid * sizeof(float));
	cudaMalloc((void**)&gav->dft.cosf_d, gav->dft.d_grid * sizeof(float));

	float* tmp = new float[gav->dft.d_grid];
	for (int i = 0; i < gav->dft.d_grid; i++) {
		tmp[i] = sinf(i / (float)gav->dft.d_grid * 2.0f * M_PI);
	}
	cudaMemcpyAsync(gav->dft.sinf_d, tmp, gav->dft.d_grid * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaStreamSynchronize(cuda_streams[0]);
	for (int i = 0; i < gav->dft.d_grid; i++) {
		tmp[i] = cosf(i / (float)gav->dft.d_grid * 2.0f * M_PI);
	}
	cudaMemcpyAsync(gav->dft.cosf_d, tmp, gav->dft.d_grid * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaStreamSynchronize(cuda_streams[0]);

	delete[] tmp;

	gav->base_c = 0.6;
	gav->base_a = 0.057;

	gav->ranges = new int[14];
	for (int i = 0; i < 7; i++) {
		gav->ranges[2 * i] = (int)(i * (gav->dft_size / 7.0));
		gav->ranges[2 * i + 1] = (int)((i+1) * (gav->dft_size / 7.0));
	}
	cudaMalloc((void**)&gav->d_ranges, 14 * sizeof(int));
	gpu_on_update_ranges(gav);

	gav->dft_out = dft_out;
	gav->gmb_in = nullptr;
	gav->vs_transition = nullptr;
	gav->vs_out = nullptr;
}

void gpu_on_update_ranges(struct gpu_audiovisual* gav) {
	cudaMemcpyAsync(gav->d_ranges, gav->ranges, 14 * sizeof(int), cudaMemcpyHostToDevice, cuda_streams[3]);
}

void gpu_audiovisual_on_input_connect(struct application_graph_node* agn, int input_id) {
	struct gpu_audiovisual* gav = (struct gpu_audiovisual*)agn->component;

	if (input_id == 2) {
		gav->transition_total = gav->vs_transition->frame_count;
	}
}

DWORD* gpu_audiovisual_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_audiovisual* gav = (struct gpu_audiovisual*)agn->component;

	if (gav->gmb_in == nullptr || gav->vs_out == nullptr) return NULL;
	int last_gpu_id = -1;

	for (int i = 0; i < gav->gmb_in->slots; i++) {
		cudaMemcpyAsync(gav->gmb_in->p_device + (i * gav->gmb_in->size), gav->mats_in[i].data, gav->gmb_in->size, cudaMemcpyHostToDevice, cuda_streams[0]);
	}
	cudaStreamSynchronize(cuda_streams[0]);
	
	int transition_total = gav->transition_total;
	bool transition_started = false;
	bool transition_switched = false;
	int transition_inactive_counter = 0;
	int last_transition_frame_id = 0;
	int next_transition_frame_id = 0;
	int transition_frame = 0;

	int last_audio_id = 0;
	int hz = (int)gav->audio_source_in->wave_format.nAvgBytesPerSec;

	while (agn->process_run) {
		int next_audio_id = 0;
		if (!gav->audio_source_in->copy_to_gmb) {
			shared_memory_buffer_try_r(gav->audio_source_in->smb, gav->audio_source_in->smb->slots, true, 8);
			next_audio_id = gav->audio_source_in->smb->p_buf_c[gav->audio_source_in->smb->slots * gav->audio_source_in->smb->size + (gav->audio_source_in->smb->slots + 1) * 2];
			shared_memory_buffer_release_r(gav->audio_source_in->smb, gav->audio_source_in->smb->slots);
		} else {
			gpu_memory_buffer_try_r(gav->audio_source_in->gmb, gav->audio_source_in->gmb->slots, true, 8);
			next_audio_id = gav->audio_source_in->gmb->p_rw[(gav->audio_source_in->gmb->slots + 1) * 2];
			gpu_memory_buffer_release_r(gav->audio_source_in->gmb, gav->audio_source_in->gmb->slots);
		}

		if (next_audio_id != last_audio_id) {
			if (!gav->audio_source_in->copy_to_gmb) {
				shared_memory_buffer_try_r(gav->audio_source_in->smb, last_audio_id, true, 8);
				shared_memory_buffer_try_r(gav->audio_source_in->smb, next_audio_id, true, 8);
			} else {
				gpu_memory_buffer_try_r(gav->audio_source_in->gmb, last_audio_id, true, 8);
				gpu_memory_buffer_try_r(gav->audio_source_in->gmb, next_audio_id, true, 8);
			}
			
			int fps = agn->process_tps_balancer.tps_target;
			for (int frame = 0; frame < fps; frame++) {
				application_graph_tps_balancer_timer_start(agn);

				float values[7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

				if (!gav->audio_source_in->copy_to_gmb) {
					unsigned char* last_audio = &gav->audio_source_in->smb->p_buf_c[last_audio_id * gav->audio_source_in->smb->size];
					unsigned char* next_audio = &gav->audio_source_in->smb->p_buf_c[next_audio_id * gav->audio_source_in->smb->size];
					unsigned char audio;
					
					int dft_N = gav->dft_size;

					for (int coeffs = 0; coeffs < dft_N; coeffs++) {
						gav->dft.norms[coeffs] = 0.0f;

						float real = 0.0f;
						float img = 0.0f;

						for (int s = 0; s < hz; s++) {
							if (s + frame * (hz / fps) < hz) {
								audio = last_audio[s + frame * (hz / fps)];
							}
							else {
								audio = next_audio[s + frame * (hz / fps) - hz];
							}
							real += ((((float)audio - 128.0f) / 128.0f) * cosf((2.0f * (float)s * (float)coeffs * 3.14159265358979f) / (float)hz)) / (float)hz;
							img += -((((float)audio - 128.0f) / 128.0f) * sinf((2.0f * (float)s * (float)coeffs * 3.14159265358979f) / (float)hz)) / (float)hz;
						}
						gav->dft.norms[coeffs] = sqrtf((real * real) + (img * img)) * gav->amplify;

						int value_idx = coeffs / (dft_N / 7);
						values[value_idx] += gav->dft.norms[coeffs] / ((float)dft_N / 7.0f);
					}
					for (int p = 0; p < 7; p++) {
						if (values[p] > 1.0f) values[p] = 1.0f;
						if (values[p] < 0.0f) values[p] = 0.0f;
					}
				} else {
					gpu_audiovisual_dft_kernel_launch(gav->audio_source_in->gmb->p_device + last_audio_id * gav->audio_source_in->gmb->size, gav->audio_source_in->gmb->p_device + next_audio_id * gav->audio_source_in->gmb->size, gav->dft_out->p_device, frame, hz, fps, gav->dft_size, gav->amplify, gav->dft.sinf_d, gav->dft.cosf_d);
					gpu_audiovisual_dft_sum_kernel_launch(gav->dft_out->p_device, gav->dft_size, gav->base_c, gav->base_a, gav->d_ranges);
				}

				if (gav->vs_transition != nullptr) {
					gpu_memory_buffer_try_r(gav->vs_transition->gmb, gav->vs_transition->gmb->slots, true, 8);
					next_transition_frame_id = gav->vs_transition->gmb->p_rw[(gav->vs_transition->gmb->slots + 1) * 2];
					gpu_memory_buffer_release_r(gav->vs_transition->gmb, gav->vs_transition->gmb->slots);
					if (next_transition_frame_id != last_transition_frame_id) {
						int transition_frame_diff = next_transition_frame_id - last_transition_frame_id;
						if (transition_frame_diff < 0) transition_frame_diff += gav->vs_transition->gmb->slots;
						transition_frame += transition_frame_diff;
						transition_inactive_counter = 0;
						transition_started = true;
						if (transition_frame > transition_total / 2 && !transition_switched) {
							gav->active_theme = (gav->active_theme + 1) % gav->theme_count;
							if (gav->active_theme == gav->transition_theme_id) gav->active_theme = (gav->active_theme + 1) % gav->theme_count;
							transition_switched = true;
						}
						last_transition_frame_id = next_transition_frame_id;
						if (transition_frame >= gav->transition_total) {
							transition_frame -= gav->transition_total;
							transition_switched = false;
						}
					} else {
						if (transition_inactive_counter < 10) {
							transition_inactive_counter++;
						} else {
							transition_frame = 0;
							transition_switched = false;
							transition_started = false;
						}
					}
				}

				gpu_memory_buffer_try_r(gav->vs_out->gmb, gav->vs_out->gmb->slots, true, 8);
				int next_gpu_out_id = (gav->vs_out->gmb->p_rw[2 * (gav->vs_out->gmb->slots + 1)] + 1) % gav->vs_out->gmb->slots;
				gpu_memory_buffer_release_r(gav->vs_out->gmb, gav->vs_out->gmb->slots);

				gpu_memory_buffer_try_rw(gav->vs_out->gmb, next_gpu_out_id, true, 8);

				const unsigned char* src = gav->gmb_in->p_device + (gav->active_theme * 9 * gav->gmb_in->size);
				const unsigned char* src_2 = gav->gmb_in->p_device + (gav->transition_theme_id * 9 * gav->gmb_in->size);
				const unsigned char* src_t = nullptr;
				if (transition_started) {
					gpu_memory_buffer_try_r(gav->vs_transition->gmb, next_transition_frame_id, true, 8);
					src_t = gav->vs_transition->gmb->p_device + (next_transition_frame_id * gav->vs_transition->gmb->size);
				}

				if (!gav->audio_source_in->copy_to_gmb) {
					gpu_audiovisual_kernel_launch(src, src_2, src_t, transition_started, transition_frame, transition_total, gav->transition_fade, gav->vs_out->gmb->p_device + (next_gpu_out_id * gav->vs_out->video_channels * gav->vs_out->video_width * gav->vs_out->video_height), gav->vs_out->video_height, gav->vs_out->video_width, 3, gav->vs_out->video_channels, false, values[0], values[1], values[2], values[3], values[4], values[5], values[6], nullptr, gav->dft_size, gav->d_ranges);
				} else {
					gpu_audiovisual_kernel_launch(src, src_2, src_t, transition_started, transition_frame, transition_total, gav->transition_fade, gav->vs_out->gmb->p_device + (next_gpu_out_id * gav->vs_out->video_channels * gav->vs_out->video_width * gav->vs_out->video_height), gav->vs_out->video_height, gav->vs_out->video_width, 3, gav->vs_out->video_channels, true, values[0], values[1], values[2], values[3], values[4], values[5], values[6], gav->dft_out->p_device, gav->dft_size, gav->d_ranges);
				}

				if (transition_started) {
					gpu_memory_buffer_release_r(gav->vs_transition->gmb, next_transition_frame_id);
				}
				gpu_memory_buffer_set_time(gav->vs_out->gmb, next_gpu_out_id, gpu_memory_buffer_get_time(gav->audio_source_in->gmb, next_audio_id));

				gpu_memory_buffer_release_rw(gav->vs_out->gmb, next_gpu_out_id);

				gpu_memory_buffer_try_rw(gav->vs_out->gmb, gav->vs_out->gmb->slots, true, 8);
				gav->vs_out->gmb->p_rw[2 * (gav->vs_out->gmb->slots + 1)] = next_gpu_out_id;
				gpu_memory_buffer_release_rw(gav->vs_out->gmb, gav->vs_out->gmb->slots);

				application_graph_tps_balancer_timer_stop(agn);
				application_graph_tps_balancer_sleep(agn);
			}

			if (!gav->audio_source_in->copy_to_gmb) {
				shared_memory_buffer_release_r(gav->audio_source_in->smb, last_audio_id);
				shared_memory_buffer_release_r(gav->audio_source_in->smb, next_audio_id);
			} else {
				gpu_memory_buffer_release_r(gav->audio_source_in->gmb, last_audio_id);
				gpu_memory_buffer_release_r(gav->audio_source_in->gmb, next_audio_id);
			}

			last_audio_id = next_audio_id;
		}
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_audiovisual_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_audiovisual* gav = (struct gpu_audiovisual*)agn->component;

	stringstream s_out;
	s_out << gav->name << std::endl;
	s_out << gav->dft_size << std::endl;

	for (int i = 0; i < 7; i++) {
		s_out << gav->ranges[2 * i] << "," << gav->ranges[(2 * i) + 1];
		if (i + 1 < 7) s_out << ",";
	}
	s_out << std::endl;

	s_out << gav->base_c << std::endl;
	s_out << gav->base_a << std::endl;
	s_out << gav->amplify << std::endl;
	s_out << gav->active_theme << std::endl;
	s_out << gav->transition_theme_id << std::endl;
	s_out << gav->transition_fade << std::endl;
	for (int i = 0; i < gav->frame_names.size(); i++) {
		s_out << gav->frame_names[i];
		if (i + 1 < gav->frame_names.size()) s_out << ",";
	}
	s_out << std::endl;

	out_str = s_out.str();
}

void gpu_audiovisual_load(struct gpu_audiovisual* gav, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	string name = line;
	std::getline(in_f, line);
	int dft_size = stoi(line.c_str());
	string ranges;
	std::getline(in_f, ranges);
	std::getline(in_f, line);
	float base_c = stof(line.c_str());
	std::getline(in_f, line);
	float base_a = stof(line.c_str());
	std::getline(in_f, line);
	gav->amplify = stof(line.c_str());
	std::getline(in_f, line);
	gav->active_theme = stoi(line.c_str());
	std::getline(in_f, line);
	gav->transition_theme_id = stoi(line.c_str());
	std::getline(in_f, line);
	gav->transition_fade = stoi(line.c_str());

	std::getline(in_f, line);
	int start = 0;
	int end = line.find_first_of(",", start);
	while (end != std::string::npos) {
		gav->frame_names.push_back(line.substr(start, end - start).c_str());
		start = end + 1;
		end = line.find_first_of(",", start);
	}
	gav->frame_names.push_back(line.substr(start, end - start).c_str());

	gpu_audiovisual_init(gav, name.c_str(), dft_size);

	gav->base_c = base_c;
	gav->base_a = base_a;

	string range_l = ranges;
	start = 0;
	end = range_l.find_first_of(",", start);
	int ct = 0;
	while (end != std::string::npos) {
		gav->ranges[ct] = stoi(range_l.substr(start, end - start).c_str());
		start = end + 1;
		end = range_l.find_first_of(",", start);
		ct++;
	}
	gav->ranges[ct] = stoi(range_l.substr(start, end - start).c_str());
}

void gpu_audiovisual_destroy(struct application_graph_node* agn) {
	/*
	struct gpu_edge_filter* gef = (struct gpu_edge_filter*)agn->component;
	delete gef;
	*/
}
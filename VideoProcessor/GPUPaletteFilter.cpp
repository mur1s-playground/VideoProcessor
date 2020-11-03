#include "GPUPaletteFilter.h"

#include "PaletteFilterKernel.h"
#include "ApplicationGraph.h"
#include "MainUI.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

#include <sstream>
#include <fstream>

#include "Logger.h"

void gpu_palette_filter_init(struct gpu_palette_filter* gpf, float palette_auto_time, int palette_auto_size, int palette_auto_bucket_count, int palette_auto_quantization_size) {
	gpf->palette_auto_time = palette_auto_time;
	gpf->palette_auto_size = palette_auto_size;
	gpf->palette_auto_bucket_count = palette_auto_bucket_count;
	gpf->palette_auto_quantization_size = palette_auto_quantization_size;
	gpf->device_palette_switch = false;

	gpf->palette_auto_timer = 0.0f;

	cudaMalloc((void**)&gpf->device_palette[!gpf->device_palette_switch], 3 * sizeof(float));
	gpf->palette_size[!gpf->device_palette_switch] = 1;

	if (gpf->palette_auto_time == 0) {
		gpf->palette_size[gpf->device_palette_switch] = gpf->palette.size() / 3;

		cudaMalloc((void**)&gpf->device_palette[gpf->device_palette_switch], 3 * gpf->palette_size[gpf->device_palette_switch] * sizeof(float));
		cudaMemcpyAsync(gpf->device_palette[gpf->device_palette_switch], gpf->palette.data(), 3 * gpf->palette_size[gpf->device_palette_switch] * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
		cudaStreamSynchronize(cuda_streams[0]);
	} else {
		gpf->palette_size[gpf->device_palette_switch] = 1;
		cudaMalloc((void**)&gpf->device_palette[gpf->device_palette_switch], 3 * sizeof(float));
	}
	
	gpf->vs_in = nullptr;
	gpf->gmb_out = nullptr;
}

void gpu_palette_filter_edit(struct gpu_palette_filter* gpf, float palette_auto_time, int palette_auto_size) {
	gpf->palette_auto_time = palette_auto_time;
	gpf->palette_auto_size = palette_auto_size;
	gpf->palette_auto_timer = 0.0f;

	if (gpf->palette_auto_time == 0) {

		gpf->palette_size[!gpf->device_palette_switch] = gpf->palette.size() / 3;

		cudaFree(gpf->device_palette[!gpf->device_palette_switch]);
		cudaMalloc((void**)&gpf->device_palette[!gpf->device_palette_switch], 3 * gpf->palette_size[!gpf->device_palette_switch] * sizeof(float));
		cudaMemcpyAsync(gpf->device_palette[!gpf->device_palette_switch], gpf->palette.data(), 3 * gpf->palette_size[!gpf->device_palette_switch] * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
		cudaStreamSynchronize(cuda_streams[0]);
		gpf->device_palette_switch = !gpf->device_palette_switch;
	} else {
		gpf->palette_auto_timer = 0.0f;
	}
}

struct pair_i_vec_less_than_key {
	inline bool operator() (const pair<int, struct vector3uc>& struct1, const pair<int, struct vector3uc>& struct2)
	{
		return (struct1.first < struct2.first);
	}
};


void gpu_palette_auto_build(struct gpu_palette_filter* mb) {
	vector<vector<pair<int, struct vector3uc>>> buckets;

	int palette_bucket_count = mb->palette_auto_bucket_count;
	int palette_bucket_quantization_size = mb->palette_auto_quantization_size;
	int palette_bucket_quantization_dim = 256 / palette_bucket_quantization_size;

	int palette_bucket_start = 0;
	int palette_bucket_stop = (int) pow(palette_bucket_quantization_dim, 3);
	int palette_bucket_dimension_size = (int)ceilf(((float)(palette_bucket_stop - palette_bucket_start))/(float)palette_bucket_count);

	for (int bc = 0; bc < palette_bucket_count; bc++) {
		buckets.push_back(vector<pair<int, struct vector3uc>>());
	}

	if (mb->vs_in->smb != nullptr) {
		shared_memory_buffer_try_r(mb->vs_in->smb, mb->vs_in->smb->slots, true, 8);
		int next_frame_id = mb->vs_in->gmb->p_rw[2 * (mb->vs_in->gmb->slots + 1)];
		shared_memory_buffer_release_r(mb->vs_in->smb, mb->vs_in->smb->slots);

		shared_memory_buffer_try_r(mb->vs_in->smb, next_frame_id, true, 8);

		unsigned char* pixels = &mb->vs_in->smb->p_buf_c[next_frame_id * mb->vs_in->video_channels * mb->vs_in->video_height * mb->vs_in->video_width];
		for (int row = 0; row < mb->vs_in->video_height; row++) {
			for (int col = 0; col < mb->vs_in->video_width; col++) {
				struct vector3uc vuc;
				vuc.r = pixels[row * mb->vs_in->video_width * mb->vs_in->video_channels + col * mb->vs_in->video_channels + 0];
				vuc.r /= palette_bucket_quantization_size;

				vuc.g = pixels[row * mb->vs_in->video_width * mb->vs_in->video_channels + col * mb->vs_in->video_channels + 1];
				vuc.g /= palette_bucket_quantization_size;

				vuc.b = pixels[row * mb->vs_in->video_width * mb->vs_in->video_channels + col * mb->vs_in->video_channels + 2];
				vuc.b /= palette_bucket_quantization_size;
				
				int bucket_number = (vuc.r * (palette_bucket_quantization_dim * palette_bucket_quantization_dim) + vuc.g * palette_bucket_quantization_dim + vuc.b) / palette_bucket_dimension_size;
				bool found = false;
				int colors = -1;
				for (; colors < buckets[bucket_number].size(); colors++) {
					struct vector3uc bvuc = buckets[bucket_number][colors].second;
					if (bvuc.r != vuc.r) continue;
					if (bvuc.g != vuc.g) continue;
					if (bvuc.b != vuc.b) continue;
					found = true;
					break;
				}
				if (found) {
					buckets[bucket_number][colors].first++;
				} else {
					buckets[bucket_number].push_back(pair<int, struct vector3uc>(1, vuc));
				}
			}
		}

		shared_memory_buffer_release_r(mb->vs_in->smb, next_frame_id);
		
		for (int bc = 0; bc < palette_bucket_count; bc++) {
			sort(buckets[bc].begin(), buckets[bc].end(), pair_i_vec_less_than_key());
		}

		mb->palette.clear();

		int colors_per_bucket = mb->palette_auto_size / palette_bucket_count;

		for (int c = 0; c < palette_bucket_count; c++) {
			int bucket_number = c;
			for (int cp = 0; cp < colors_per_bucket; cp++) {
				int idx = buckets[bucket_number].size() - 1 - cp;
				if (idx >= 0) {
					struct vector3uc color = buckets[bucket_number][idx].second;
					mb->palette.push_back(color.r * palette_bucket_quantization_size);
					mb->palette.push_back(color.g * palette_bucket_quantization_size);
					mb->palette.push_back(color.b * palette_bucket_quantization_size);
				}
			}
		}

		mb->palette_size[!mb->device_palette_switch] = mb->palette.size() / 3;

		cudaFree(mb->device_palette[!mb->device_palette_switch]);
		cudaMalloc((void**)&mb->device_palette[!mb->device_palette_switch], 3 * mb->palette_size[!mb->device_palette_switch] * sizeof(float));
		cudaMemcpyAsync(mb->device_palette[!mb->device_palette_switch], mb->palette.data(), 3 * mb->palette_size[!mb->device_palette_switch] * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
		cudaStreamSynchronize(cuda_streams[0]);
		mb->device_palette_switch = !mb->device_palette_switch;
	}

}

DWORD* gpu_palette_filter_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_palette_filter* mb = (struct gpu_palette_filter*)agn->component;

	if (mb->vs_in == nullptr || mb->vs_in->gmb == nullptr) return NULL;
	int last_gpu_id = -1;

	while (agn->process_run) {
		if (mb->palette_auto_timer <= 0.0f) {
			gpu_palette_auto_build(mb);
			mb->palette_auto_timer = mb->palette_auto_time;
		} else {
			mb->palette_auto_timer -= (33.0f / 1800000.0f);
		}

		application_graph_tps_balancer_timer_start(agn);

		gpu_memory_buffer_try_r(mb->vs_in->gmb, mb->vs_in->gmb->slots, true, 8);
		int next_gpu_id = mb->vs_in->gmb->p_rw[2 * (mb->vs_in->gmb->slots + 1)];
		gpu_memory_buffer_release_r(mb->vs_in->gmb, mb->vs_in->gmb->slots);
		if (next_gpu_id != last_gpu_id) {
			gpu_memory_buffer_try_r(mb->gmb_out, mb->gmb_out->slots, true, 8);
			int next_gpu_out_id = (mb->gmb_out->p_rw[2 * (mb->gmb_out->slots + 1)] + 1) % mb->gmb_out->slots;
			gpu_memory_buffer_release_r(mb->gmb_out, mb->gmb_out->slots);

			gpu_memory_buffer_try_rw(mb->gmb_out, next_gpu_out_id, true, 8);

			gpu_memory_buffer_try_r(mb->vs_in->gmb, next_gpu_id, true, 8);

			palette_filter_kernel_launch(mb->vs_in->gmb->p_device + (next_gpu_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels), mb->gmb_out->p_device + (next_gpu_out_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels), mb->vs_in->video_width, mb->vs_in->video_height, mb->vs_in->video_channels, mb->device_palette[mb->device_palette_switch], mb->palette_size[mb->device_palette_switch]);

			gpu_memory_buffer_release_r(mb->vs_in->gmb, next_gpu_id);

			gpu_memory_buffer_release_rw(mb->gmb_out, next_gpu_out_id);

			gpu_memory_buffer_try_rw(mb->gmb_out, mb->gmb_out->slots, true, 8);
			mb->gmb_out->p_rw[2 * (mb->gmb_out->slots + 1)] = next_gpu_out_id;
			gpu_memory_buffer_release_rw(mb->gmb_out, mb->gmb_out->slots);
			last_gpu_id = next_gpu_id;
		}
		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_palette_filter_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_palette_filter* mb = (struct gpu_palette_filter*)agn->component;

	stringstream s_out;
	s_out << mb->palette_auto_time << std::endl;
	s_out << mb->palette_auto_size << std::endl;
	s_out << mb->palette_auto_bucket_count << std::endl;
	s_out << mb->palette_auto_quantization_size << std::endl;

	for (int i = 0; i < mb->palette.size(); i++) {
		if (i > 0) {
			s_out << ",";
		}
		s_out << mb->palette[i];
	}
	out_str = s_out.str();
}

void gpu_palette_filter_load(struct gpu_palette_filter* gb, ifstream& in_f) {
	std::string line;

	std::getline(in_f, line);
	gb->palette_auto_time = stof(line);
	std::getline(in_f, line);
	gb->palette_auto_size = stoi(line);
	std::getline(in_f, line);
	gb->palette_auto_bucket_count = stoi(line);
	std::getline(in_f, line);
	gb->palette_auto_quantization_size = stoi(line);


	std::getline(in_f, line);
	int start = 0;
	int end = line.find_first_of(",", start);
	while (end != std::string::npos) {
		gb->palette.push_back(stof(line.substr(start, end - start).c_str()));
		start = end + 1;
		end = line.find_first_of(",", start);
	}
	gb->palette.push_back(stof(line.substr(start, end - start).c_str()));

	gpu_palette_filter_init(gb, gb->palette_auto_time, gb->palette_auto_size, gb->palette_auto_bucket_count, gb->palette_auto_quantization_size);
}

void gpu_palette_filter_destroy(struct application_graph_node* agn) {
	struct gpu_palette_filter* mb = (struct gpu_palette_filter*)agn->component;

	cudaFree(mb->device_palette);
	delete mb;
}
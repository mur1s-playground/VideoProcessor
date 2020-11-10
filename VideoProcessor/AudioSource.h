#pragma once

#include "Windows.h"
#include "SharedMemoryBuffer.h"
#include "GPUMemoryBuffer.h"

struct audio_source {
	int device_id;
	bool copy_to_gmb;

	HWAVEIN wave_in_handle;
	WAVEFORMATEX wave_format;

	WAVEHDR* wave_header_arr;

	MMRESULT wave_status;

	struct shared_memory_buffer* smb;
	int smb_last_used_id;
	int smb_size_req;

	struct gpu_memory_buffer* gmb;
};

void audio_source_init(struct audio_source* vs, int device_id, int channels, int samples_per_sec, int bits_per_sample, bool copy_to_gmb);

void audio_source_on_input_connect(struct application_graph_node* agn, int input_id);
DWORD* audio_source_loop(LPVOID args);

void audio_source_externalise(struct application_graph_node* agn, string& out_str);
void audio_source_load(struct audio_source* as, ifstream& in_f);
void audio_source_destory(struct application_graph_node* agn);
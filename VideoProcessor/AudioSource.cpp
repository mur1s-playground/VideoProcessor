#include "AudioSource.h"

#include "ApplicationGraph.h"

#include "Logger.h"

#include <sstream>
#include <fstream>

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

void CALLBACK audio_source_process_callback(
	HWAVEIN   hwi,
	UINT      uMsg,
	DWORD_PTR dwInstance,
	DWORD_PTR dwParam1,
	DWORD_PTR dwParam2) {
	if (uMsg == WIM_CLOSE) {
		
	} else if (uMsg == WIM_DATA) {

	} else if (uMsg == WIM_CLOSE) {

	}
}

void audio_source_init(struct audio_source* as, int device_id, int channels, int samples_per_sec, int bits_per_sample, bool copy_to_gmb) {
	as->device_id = device_id;
	as->copy_to_gmb = copy_to_gmb;

	as->wave_format.wFormatTag = WAVE_FORMAT_PCM;
	as->wave_format.nChannels = channels;
	as->wave_format.nSamplesPerSec = samples_per_sec;
	as->wave_format.wBitsPerSample = bits_per_sample;
	as->wave_format.nBlockAlign = (channels * bits_per_sample) / 8;
	as->wave_format.nAvgBytesPerSec = channels * samples_per_sec;
	as->wave_format.cbSize = 0;

	as->smb_size_req = channels * bits_per_sample * samples_per_sec;

	as->wave_status = waveInOpen(&as->wave_in_handle, device_id, &as->wave_format, (DWORD_PTR)&audio_source_process_callback, (DWORD_PTR)(void *)as, CALLBACK_FUNCTION);

	if (as->wave_status != MMSYSERR_NOERROR) {

	} else {

	}
}

void audio_source_prepare_hdr(struct audio_source *as, int id) {
	as->wave_header_arr[id].lpData = (LPSTR)&as->smb->p_buf_c[id * as->smb->size];
	as->wave_header_arr[id].dwBufferLength = as->wave_format.nAvgBytesPerSec;
	as->wave_header_arr[id].dwBytesRecorded = 0;
	as->wave_header_arr[id].dwUser = 0L;
	as->wave_header_arr[id].dwFlags = 0L;
	as->wave_header_arr[id].dwLoops = 0L;

	MMRESULT rc = waveInPrepareHeader(as->wave_in_handle, &as->wave_header_arr[id], sizeof(WAVEHDR));

	if (rc != MMSYSERR_NOERROR) {
		as->wave_status = rc;
	} else {
		rc = waveInAddBuffer(as->wave_in_handle, &as->wave_header_arr[id], sizeof(WAVEHDR));
	}
}

void audio_source_on_input_connect(struct application_graph_node* agn, int input_id) {
	if (input_id == 0) {
		struct audio_source* as = (struct audio_source*)agn->component;
		as->wave_header_arr = new WAVEHDR[as->smb->slots];
		for (int wh = 0; wh < as->smb->slots; wh++) {
			audio_source_prepare_hdr(as, wh);
		}
	}
}

DWORD* audio_source_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct audio_source* as = (struct audio_source*)agn->component;

	if (as->smb == nullptr) return NULL;

	shared_memory_buffer_try_rw(as->smb, 0, true, 8);
	as->smb_last_used_id = 0;

	as->wave_status = waveInStart(as->wave_in_handle);

	while (as->wave_status == MMSYSERR_NOERROR && agn->process_run) {
		do {
			Sleep(33);
		} while (waveInUnprepareHeader(as->wave_in_handle, &as->wave_header_arr[as->smb_last_used_id], sizeof(WAVEHDR)) == WAVERR_STILLPLAYING);
		shared_memory_buffer_set_time(as->smb, as->smb_last_used_id, application_graph_tps_balancer_get_time());

		if (as->copy_to_gmb) {
			gpu_memory_buffer_try_r(as->gmb, as->gmb->slots, true, 8);
			int next_gpu_id = (as->gmb->p_rw[2 * (as->gmb->slots + 1)] + 1) % as->gmb->slots;
			gpu_memory_buffer_release_r(as->gmb, as->gmb->slots);
			gpu_memory_buffer_try_rw(as->gmb, next_gpu_id, true, 8);
			cudaMemcpyAsync(as->gmb->p_device + (next_gpu_id * as->gmb->size), &as->smb->p_buf_c[as->smb_last_used_id * as->smb->size], as->smb->size, cudaMemcpyHostToDevice, cuda_streams[0]);
			cudaStreamSynchronize(cuda_streams[0]);
			gpu_memory_buffer_set_time(as->gmb, next_gpu_id, shared_memory_buffer_get_time(as->smb, as->smb_last_used_id));
			gpu_memory_buffer_release_rw(as->gmb, next_gpu_id);
			gpu_memory_buffer_try_rw(as->gmb, as->gmb->slots, true, 8);
			as->gmb->p_rw[2 * (as->gmb->slots + 1)] = next_gpu_id;
			gpu_memory_buffer_release_rw(as->gmb, as->gmb->slots);
		}
		shared_memory_buffer_release_rw(as->smb, as->smb_last_used_id);

		as->wave_status = waveInPrepareHeader(as->wave_in_handle, &as->wave_header_arr[as->smb_last_used_id], sizeof(WAVEHDR));

		if (as->wave_status == MMSYSERR_NOERROR) {
			as->wave_status = waveInAddBuffer(as->wave_in_handle, &as->wave_header_arr[as->smb_last_used_id], sizeof(WAVEHDR));
		}

		shared_memory_buffer_try_rw(as->smb, as->smb->slots, true, 8);
		as->smb->p_buf_c[as->smb->slots * as->smb->size +			((as->smb->slots + 1) * 2)] = as->smb_last_used_id;
		shared_memory_buffer_release_rw(as->smb, as->smb->slots);
		as->smb_last_used_id = (as->smb_last_used_id + 1) % as->smb->slots;

		shared_memory_buffer_try_rw(as->smb, as->smb_last_used_id, true, 8);
	}

	shared_memory_buffer_release_rw(as->smb, as->smb_last_used_id);

	agn->process_run = false;
}

void audio_source_externalise(struct application_graph_node* agn, string& out_str) {
	struct audio_source* as = (struct audio_source*)agn->component;

	stringstream s_out;
	s_out << as->device_id << std::endl;
	s_out << as->wave_format.nChannels << std::endl;
	s_out << as->wave_format.nSamplesPerSec << std::endl;
	s_out << as->wave_format.wBitsPerSample << std::endl;
	if (as->copy_to_gmb) {
		s_out << 1 << std::endl;
	} else {
		s_out << 0 << std::endl;
	}

	out_str = s_out.str();
}

void audio_source_load(struct audio_source* as, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	int device_id = stoi(line);
	std::getline(in_f, line);
	int channels = stoi(line);
	std::getline(in_f, line);
	int samples_per_sec = stoi(line);
	std::getline(in_f, line);
	int bits_per_sample = stoi(line);
	std::getline(in_f, line);
	bool copy_to_gmb = stoi(line) == 1;

	audio_source_init(as, device_id, channels, samples_per_sec, bits_per_sample, copy_to_gmb);
}

//TODO: complete
void audio_source_destory(struct application_graph_node* agn) {
	struct audio_source* as = (struct audio_source*)agn->component;
	delete as;
}
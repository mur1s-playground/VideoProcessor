#include "VideoSource.h"

#include "cuda_runtime.h"
#include <string>

#include "ApplicationGraph.h"
#include "MainUI.h"

#include "CUDAStreamHandler.h"

#include "Logger.h"

void video_source_set_meta(struct video_source* vs) {
	if (vs->is_open) {
		vs->video_width = vs->video_capture.get(cv::CAP_PROP_FRAME_WIDTH);
		vs->video_height = vs->video_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
		vs->video_channels = 3;
		vs->smb_size_req = vs->video_width * vs->video_height * vs->video_channels;
	}
	vs->smb = nullptr;
	vs->gmb = nullptr;
	vs->mats = nullptr;
}

void video_source_init(struct video_source* vs, int device_id) {
	stringstream ss_name;
	ss_name << device_id;
	vs->name = ss_name.str();

	vs->video_capture.open(device_id);
	vs->is_open = vs->video_capture.isOpened();
	video_source_set_meta(vs);
}

void video_source_init(struct video_source* vs, const char* path) {
	stringstream ss_name;
	ss_name << path;
	vs->name = ss_name.str();

	const char* str = vs->name.c_str();
	if (strstr(str, "dummy") == str) {
		vs->read_video_capture = false;
	} else {
		vs->read_video_capture = true;
		vs->video_capture.open(path);
		vs->is_open = vs->video_capture.isOpened();
		video_source_set_meta(vs);
	}
}

void video_source_on_input_connect(struct application_graph_node *agn, int input_id) {
	if (input_id == 0) {
		struct video_source* vs = (struct video_source*)agn->component;

		if (vs->mats == nullptr) {
			vs->mats = new Mat[vs->smb->slots];
			for (int i = 0; i < vs->smb->slots; i++) {
				if (vs->video_channels == 1) {
					vs->mats[i] = Mat(vs->video_height, vs->video_width, CV_8UC1, &vs->smb->p_buf_c[i * vs->video_channels * vs->video_height * vs->video_width]);
				}
				else if (vs->video_channels == 3) {
					vs->mats[i] = Mat(vs->video_height, vs->video_width, CV_8UC3, &vs->smb->p_buf_c[i * vs->video_channels * vs->video_height * vs->video_width]);
				}
				else if (vs->video_channels == 4) {
					vs->mats[i] = Mat(vs->video_height, vs->video_width, CV_8UC4, &vs->smb->p_buf_c[i * vs->video_channels * vs->video_height * vs->video_width]);
				}
			}
			vs->smb_last_used_id = vs->smb->slots - 1;
			vs->smb_framecount = vs->smb->slots;
		}
	}
}

DWORD* video_source_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct video_source* vs = (struct video_source*)agn->component;

	if (vs->smb == nullptr) return NULL;

	if (vs->direction_smb_to_gmb) {
			bool run = true;
			int last_id = -1;
			while (agn->process_run) {
				int next_id = -1;
				if (vs->read_video_capture) {
					next_id = (vs->smb_last_used_id + 1) % vs->smb_framecount;
					shared_memory_buffer_try_rw(vs->smb, next_id, true, 8);
					vs->video_capture >> vs->mats[next_id];
					shared_memory_buffer_release_rw(vs->smb, next_id);
					if (vs->mats[next_id].empty()) {
						vs->is_open = false;
						run = false;
						break;
					}
				} else {
					shared_memory_buffer_try_r(vs->smb, vs->smb_framecount, true, 8);
					next_id = vs->smb->p_buf_c[vs->smb_framecount * vs->video_channels * vs->video_height * vs->video_width + ((vs->smb_framecount + 1) * 2)];
					shared_memory_buffer_release_r(vs->smb, vs->smb_framecount);
				}
				if (last_id != next_id) {
					if (vs->do_copy) {
						gpu_memory_buffer_try_r(vs->gmb, vs->gmb->slots, true, 8);
						int next_gpu_id = (vs->gmb->p_rw[2 * (vs->gmb->slots + 1)] + 1) % vs->gmb->slots;
						gpu_memory_buffer_release_r(vs->gmb, vs->gmb->slots);
						gpu_memory_buffer_try_rw(vs->gmb, next_gpu_id, true, 8);
						shared_memory_buffer_try_r(vs->smb, next_id, true, 8);
						cudaMemcpyAsync(&vs->gmb->p_device[next_gpu_id * vs->video_channels * vs->video_height * vs->video_width], &vs->smb->p_buf_c[next_id * vs->video_channels * vs->video_height * vs->video_width], vs->video_channels * vs->video_height * vs->video_width, cudaMemcpyHostToDevice, cuda_streams[0]);
						cudaStreamSynchronize(cuda_streams[0]);
						shared_memory_buffer_release_r(vs->smb, next_id);
						gpu_memory_buffer_release_rw(vs->gmb, next_gpu_id);
						gpu_memory_buffer_try_rw(vs->gmb, vs->gmb->slots, true, 8);
						vs->gmb->p_rw[2 * (vs->gmb->slots + 1)] = next_gpu_id;
						gpu_memory_buffer_release_rw(vs->gmb, vs->gmb->slots);
					}
					shared_memory_buffer_try_rw(vs->smb, vs->smb_framecount, true, 8);
					//slots													   //rw-locks									   //meta
					vs->smb->p_buf_c[vs->smb_framecount * vs->video_channels * vs->video_height * vs->video_width + ((vs->smb_framecount + 1) * 2)] = next_id;
					shared_memory_buffer_release_rw(vs->smb, vs->smb_framecount);
					last_id = next_id;
					vs->smb_last_used_id = next_id;
				} else {
					Sleep(8);
				}
			}
	} else if (!vs->read_video_capture && vs->do_copy && !vs->direction_smb_to_gmb) {
		int last_gpu_id = -1;
		while (agn->process_run) {
			gpu_memory_buffer_try_r(vs->gmb, vs->gmb->slots, true, 8);
			int next_gpu_id = vs->gmb->p_rw[2 * (vs->gmb->slots + 1)];
			gpu_memory_buffer_release_r(vs->gmb, vs->gmb->slots);
			if (next_gpu_id != last_gpu_id) {
				gpu_memory_buffer_try_r(vs->gmb, next_gpu_id, true, 8);
				int next_id = (vs->smb_last_used_id + 1) % vs->smb_framecount;
				shared_memory_buffer_try_rw(vs->smb, next_id, true, 8);
				cudaMemcpyAsync(&vs->smb->p_buf_c[next_id * vs->video_channels * vs->video_height * vs->video_width], &vs->gmb->p_device[next_gpu_id * vs->video_channels * vs->video_height * vs->video_width], vs->video_channels * vs->video_height * vs->video_width, cudaMemcpyDeviceToHost, cuda_streams[4]);
				cudaStreamSynchronize(cuda_streams[4]);
				shared_memory_buffer_release_rw(vs->smb, next_id);
				gpu_memory_buffer_release_r(vs->gmb, next_gpu_id);
				shared_memory_buffer_try_rw(vs->smb, vs->smb_framecount, true, 8);
				vs->smb->p_buf_c[vs->smb_framecount * vs->video_channels * vs->video_height * vs->video_width + ((vs->smb_framecount + 1) * 2)] = next_id;
				shared_memory_buffer_release_rw(vs->smb, vs->smb_framecount);
				vs->smb_last_used_id = next_id;
				last_gpu_id = next_gpu_id;
			}
		}
	}
	
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}


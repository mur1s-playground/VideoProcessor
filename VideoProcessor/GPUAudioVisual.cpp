#include "AudioVisualKernel.h"

#include "GPUAudioVisual.h"

#include "CUDAStreamHandler.h"
#include "cuda_runtime.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "MainUI.h"
#include "ApplicationGraph.h"

#include <sstream>
#include <fstream>

void gpu_audiovisual_init(struct gpu_audiovisual* gav, const char *name, vector<string> files_names) {
	gav->name = string(name);

	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_0000000.png");
	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_1000000.png");
	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_0100000.png");
	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_0010000.png");
	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_0001000.png");
	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_0000100.png");
	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_0000010.png");
	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_0000001.png");
	gav->frame_names.push_back("C:\\Users\\mur1_\\Desktop\\musicvis\\untitled_1111111.png");
	/*
	for (int i = 0; i < files_names.size(); i++) {
		gav->frame_names.push_back(files_names[i]);
	}
	*/

	struct shared_memory_buffer* smb_in = new shared_memory_buffer();

	shared_memory_buffer_init(smb_in, name, 7 * sizeof(float), 30, sizeof(int));

	gav->mats_in = new Mat[9];

	for (int i = 0; i < 9; i++) {
		gav->mats_in[i] = cv::imread(gav->frame_names[i], IMREAD_COLOR);
		if (gav->mats_in[i].empty()) {
			std::cout << "Could not read the image: " << gav->frame_names[i] << std::endl;
			exit(0);
		}
	}

	gav->smb_in = smb_in;
	gav->gmb_in = nullptr;
	gav->vs_out = nullptr;
}

DWORD* gpu_audiovisual_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_audiovisual* gav = (struct gpu_audiovisual*)agn->component;

	if (gav->gmb_in == nullptr || gav->vs_out == nullptr) return NULL;
	int last_gpu_id = -1;

	for (int i = 0; i < 9; i++) {
		cudaMemcpyAsync(gav->gmb_in->p_device + (i * 3 * 1080 * 1920), gav->mats_in[i].data, 3 * 1080 * 1920, cudaMemcpyHostToDevice, cuda_streams[0]);
	}
	cudaStreamSynchronize(cuda_streams[0]);

	while (agn->process_run) {
		application_graph_tps_balancer_timer_start(agn);

	/*	gpu_memory_buffer_try_r(gav->vs_in->gmb, gef->vs_in->gmb->slots, true, 8);
		int next_gpu_id = gef->vs_in->gmb->p_rw[2 * (gef->vs_in->gmb->slots + 1)];
		gpu_memory_buffer_release_r(gef->vs_in->gmb, gef->vs_in->gmb->slots);
		if (next_gpu_id != last_gpu_id) {
		*/
			gpu_memory_buffer_try_r(gav->vs_out->gmb, gav->vs_out->gmb->slots, true, 8);
			int next_gpu_out_id = (gav->vs_out->gmb->p_rw[2 * (gav->vs_out->gmb->slots + 1)] + 1) % gav->vs_out->gmb->slots;
			gpu_memory_buffer_release_r(gav->vs_out->gmb, gav->vs_out->gmb->slots);

			//gpu_memory_buffer_try_r(gav->vs_in->gmb, next_gpu_id, true, 8);
			gpu_memory_buffer_try_rw(gav->vs_out->gmb, next_gpu_out_id, true, 8);

			float values[7] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
			
			gpu_audiovisual_kernel_launch(gav->gmb_in->p_device, gav->vs_out->gmb->p_device + (next_gpu_out_id * 3 * 1920 * 1080), 1080, 1920, 3, std::rand() /(float) RAND_MAX, std::rand() / (float)RAND_MAX, std::rand() / (float)RAND_MAX, std::rand() / (float)RAND_MAX, std::rand() / (float)RAND_MAX, std::rand() / (float)RAND_MAX, std::rand() / (float)RAND_MAX);

			gpu_memory_buffer_set_time(gav->vs_out->gmb, next_gpu_out_id, application_graph_tps_balancer_get_time());

			gpu_memory_buffer_release_rw(gav->vs_out->gmb, next_gpu_out_id);
//			gpu_memory_buffer_release_r(gef->vs_in->gmb, next_gpu_id);

			gpu_memory_buffer_try_rw(gav->vs_out->gmb, gav->vs_out->gmb->slots, true, 8);
			gav->vs_out->gmb->p_rw[2 * (gav->vs_out->gmb->slots + 1)] = next_gpu_out_id;
			gpu_memory_buffer_release_rw(gav->vs_out->gmb, gav->vs_out->gmb->slots);
			//last_gpu_id = next_gpu_id;
		//}
		application_graph_tps_balancer_timer_stop(agn);
		application_graph_tps_balancer_sleep(agn);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_audiovisual_externalise(struct application_graph_node* agn, string& out_str) {
	/*
	struct gpu_edge_filter* gef = (struct gpu_edge_filter*)agn->component;

	stringstream s_out;
	s_out << gef->amplify << std::endl;

	out_str = s_out.str();
	*/
}

void gpu_audiovisual_load(struct gpu_edge_filter* gef, ifstream& in_f) {
	/*
	std::string line;
	std::getline(in_f, line);
	gef->amplify = stof(line);

	gef->vs_in = nullptr;
	gef->gmb_out = nullptr;
	*/
}

void gpu_audiovisual_destroy(struct application_graph_node* agn) {
	/*
	struct gpu_edge_filter* gef = (struct gpu_edge_filter*)agn->component;
	delete gef;
	*/
}
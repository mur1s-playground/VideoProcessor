#include "GPUGaussianBlur.h"

#include "GaussianBlur.h"
#include "ApplicationGraph.h"
#include "MainUI.h"

#include "cuda_runtime.h"
#include "CUDAStreamHandler.h"

#include <sstream>
#include <fstream>

#include "Logger.h"

void gpu_gaussian_blur_init(struct gpu_gaussian_blur* gb, const int kernel_size, const float a, const float b, const float c) {
	gb->kernel_size = kernel_size;
	gb->a = a;
	gb->b = b;
	gb->c = c;

	gb->vs_in = nullptr;
	gb->gmb_out = nullptr;

	float* host_kernel_out = nullptr;
	gaussian_blur_construct_kernel(&host_kernel_out, &gb->norm_kernel, gb->kernel_size, gb->a, gb->b, gb->c);

	cudaMalloc((void**)&gb->device_kernel, kernel_size * kernel_size * sizeof(float));
	cudaMemcpyAsync(gb->device_kernel, host_kernel_out, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaStreamSynchronize(cuda_streams[0]);
	delete host_kernel_out;
}

void gpu_gaussian_blur_edit(struct gpu_gaussian_blur *gb, const int kernel_size, const float a, const float b, const float c) {
	gb->kernel_size = kernel_size;
	gb->a = a;
	gb->b = b;
	gb->c = c;

	float* host_kernel_out = nullptr;
	gaussian_blur_construct_kernel(&host_kernel_out, &gb->norm_kernel, gb->kernel_size, gb->a, gb->b, gb->c);

	cudaFree(gb->device_kernel);
	cudaMalloc((void**)&gb->device_kernel, kernel_size * kernel_size * sizeof(float));

	cudaMemcpyAsync(gb->device_kernel, host_kernel_out, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[0]);
	cudaStreamSynchronize(cuda_streams[0]);
	delete host_kernel_out;
}

DWORD* gpu_gaussian_blur_loop(LPVOID args) {
	struct application_graph_node* agn = (struct application_graph_node*)args;
	struct gpu_gaussian_blur* mb = (struct gpu_gaussian_blur*)agn->component;

	if (mb->vs_in == nullptr || mb->vs_in->gmb == nullptr) return NULL;
	int last_gpu_id = -1;
	while (agn->process_run) {
		gpu_memory_buffer_try_r(mb->vs_in->gmb, mb->vs_in->gmb->slots, true, 8);
		int next_gpu_id = mb->vs_in->gmb->p_rw[2 * (mb->vs_in->gmb->slots + 1)];
		gpu_memory_buffer_release_r(mb->vs_in->gmb, mb->vs_in->gmb->slots);
		if (next_gpu_id != last_gpu_id) {
			gpu_memory_buffer_try_r(mb->gmb_out, mb->gmb_out->slots, true, 8);
			int next_gpu_out_id = (mb->gmb_out->p_rw[2 * (mb->gmb_out->slots + 1)] + 1) % mb->gmb_out->slots;
			gpu_memory_buffer_release_r(mb->gmb_out, mb->gmb_out->slots);

			gpu_memory_buffer_try_rw(mb->gmb_out, next_gpu_out_id, true, 8);

			gpu_memory_buffer_try_r(mb->vs_in->gmb, next_gpu_id, true, 8);

			gaussian_blur_kernel_launch(mb->vs_in->gmb->p_device + (next_gpu_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels), mb->gmb_out->p_device + (next_gpu_out_id * mb->vs_in->video_width * mb->vs_in->video_height * mb->vs_in->video_channels), mb->vs_in->video_width, mb->vs_in->video_height, mb->vs_in->video_channels, mb->kernel_size, mb->device_kernel, mb->norm_kernel);

			gpu_memory_buffer_release_r(mb->vs_in->gmb, next_gpu_id);
			
			gpu_memory_buffer_release_rw(mb->gmb_out, next_gpu_out_id);

			gpu_memory_buffer_try_rw(mb->gmb_out, mb->gmb_out->slots, true, 8);
			mb->gmb_out->p_rw[2 * (mb->gmb_out->slots + 1)] = next_gpu_out_id;
			gpu_memory_buffer_release_rw(mb->gmb_out, mb->gmb_out->slots);
			last_gpu_id = next_gpu_id;
		}
		Sleep(16);
	}
	agn->process_run = false;
	myApp->drawPane->Refresh();
	return NULL;
}

void gpu_gaussian_blur_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_gaussian_blur* mb = (struct gpu_gaussian_blur*)agn->component;

	stringstream s_out;
	s_out << mb->kernel_size << std::endl;
	s_out << mb->a << std::endl;
	s_out << mb->b << std::endl;
	s_out << mb->c << std::endl;

	out_str = s_out.str();
}

void gpu_gaussian_blur_load(struct gpu_gaussian_blur* gb, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	gb->kernel_size = stoi(line);
	gb->a = stof(line);
	gb->b = stof(line);
	gb->c = stof(line);

	gpu_gaussian_blur_init(gb, gb->kernel_size, gb->a, gb->b, gb->c);
}

void gpu_gaussian_blur_destroy(struct application_graph_node* agn) {
	struct gpu_gaussian_blur* mb = (struct gpu_gaussian_blur*)agn->component;
	
	cudaFree(mb->device_kernel);
	delete mb;
}
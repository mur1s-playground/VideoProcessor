#include "GPUMemoryBuffer.h"

#include "cuda_runtime.h"
#include "Logger.h"
#include "ApplicationGraph.h"

#include <sstream>
#include <fstream>

void gpu_memory_buffer_init(struct gpu_memory_buffer* gmb, const char* name, int size, int slots, int meta_size) {
	gmb->name = name;
	gmb->size = size;
	gmb->slots = slots;
	gmb->meta_size = meta_size;

	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void**)&gmb->p_device, size * slots);

	if (err != cudaSuccess) {
		gmb->error = true;
	}

	gmb->p_rw = new unsigned char[(slots+1) *2 + meta_size + slots * sizeof(unsigned long long)];
	memset(gmb->p_rw, 0, (slots+1) * 2 + meta_size + slots * sizeof(unsigned long long));
	
	gmb->h_mutex = CreateMutex(NULL, FALSE, NULL);
}

void gpu_memory_buffer_edit(struct gpu_memory_buffer* gmb, const char* name, int size, int slots, int meta_size) {
	gmb->name = name;
	gmb->size = size;
	gmb->slots = slots;
	gmb->meta_size = meta_size;

	cudaFree((void **)&gmb->p_device);
	cudaMalloc((void**)&gmb->p_device, size * slots);

	delete(gmb->p_rw);
	gmb->p_rw = new unsigned char[(slots + 1) * 2 + meta_size + slots*sizeof(unsigned long long)];
	memset(gmb->p_rw, 0, (slots + 1) * 2 + meta_size + slots * sizeof(unsigned long long));
}

void gpu_memory_buffer_set_time(struct gpu_memory_buffer* gmb, int slot, unsigned long long time) {
	unsigned long long* time_pos = (unsigned long long*) &gmb->p_rw[2 * (gmb->slots + 1) + gmb->meta_size + slot * sizeof(unsigned long long)];
	*time_pos = time;
}

unsigned long long gpu_memory_buffer_get_time(struct gpu_memory_buffer* gmb, int slot) {
	unsigned long long* time_pos = (unsigned long long*) &gmb->p_rw[2 * (gmb->slots + 1) + gmb->meta_size + slot * sizeof(unsigned long long)];
	return *time_pos;
}

bool gpu_memory_buffer_try_rw(struct gpu_memory_buffer* gmb, int slot, bool block, int sleep_ms) {
	WaitForSingleObject(gmb->h_mutex, INFINITE);
	bool is_written_to = (gmb->p_rw[2 * slot + 1] == 1);
	bool is_read_from = (gmb->p_rw[2 * slot] > 0);
	bool set_write = false;
	if (block) {
		while (true) {
			if (!is_written_to) {
				gmb->p_rw[2 * slot + 1] = 1;
				set_write = true;
			}
			if (!is_read_from && set_write) {
				ReleaseMutex(gmb->h_mutex);
				return true;
			}
			ReleaseMutex(gmb->h_mutex);
			Sleep(sleep_ms);
			WaitForSingleObject(gmb->h_mutex, INFINITE);
			is_written_to = (gmb->p_rw[2 * slot + 1] == 1);
			is_read_from = (gmb->p_rw[2 * slot] > 0);
		}
	} else {
		if (!is_written_to && !is_read_from) {
			gmb->p_rw[2 * slot + 1] = 1;
			ReleaseMutex(gmb->h_mutex);
			return true;
		}
	}
	ReleaseMutex(gmb->h_mutex);
	return false;
}

void gpu_memory_buffer_release_rw(struct gpu_memory_buffer* gmb, int slot) {
	WaitForSingleObject(gmb->h_mutex, INFINITE);
	gmb->p_rw[2 * slot + 1] = 0;
	ReleaseMutex(gmb->h_mutex);
}

bool gpu_memory_buffer_try_r(struct gpu_memory_buffer* gmb, int slot, bool block, int sleep_ms) {
	WaitForSingleObject(gmb->h_mutex, INFINITE);
	bool is_written_to = (gmb->p_rw[2 * slot + 1] == 1);
	if (block) {
		while (true) {
			if (!is_written_to) {
				gmb->p_rw[2 * slot]++;
				ReleaseMutex(gmb->h_mutex);
				return true;
			}
			ReleaseMutex(gmb->h_mutex);
			Sleep(sleep_ms);
			WaitForSingleObject(gmb->h_mutex, INFINITE);
			is_written_to = (gmb->p_rw[2 * slot + 1] == 1);
		}
	}
	else {
		if (!is_written_to) {
			gmb->p_rw[2 * slot]++;
			ReleaseMutex(gmb->h_mutex);
			return true;
		}
	}
	ReleaseMutex(gmb->h_mutex);
	return false;
}

void gpu_memory_buffer_release_r(struct gpu_memory_buffer* gmb, int slot) {
	WaitForSingleObject(gmb->h_mutex, INFINITE);
	gmb->p_rw[2 * slot]--;
	ReleaseMutex(gmb->h_mutex);
}

void gpu_memory_buffer_externalise(struct application_graph_node* agn, string& out_str) {
	struct gpu_memory_buffer* gmb = (struct gpu_memory_buffer*)agn->component;

	stringstream s_out;
	s_out << gmb->name << std::endl;
	s_out << gmb->size << std::endl;
	s_out << gmb->slots << std::endl;
	s_out << gmb->meta_size << std::endl;

	out_str = s_out.str();
}

void gpu_memory_buffer_load(struct gpu_memory_buffer* gmb, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	string name = line;
	std::getline(in_f, line);
	int size = stoi(line);
	std::getline(in_f, line);
	int slots = stoi(line);
	std::getline(in_f, line);
	int meta_size = stoi(line);
	
	gpu_memory_buffer_init(gmb, name.c_str(), size, slots, meta_size);
}

void gpu_memory_buffer_destroy(struct application_graph_node* agn) {
	struct gpu_memory_buffer* gmb = (struct gpu_memory_buffer*)agn->component;
	cudaFree((void**)&gmb->p_device);
	delete(gmb->p_rw);
	CloseHandle(gmb->h_mutex);
	delete(gmb);
}
#include "GPUMemoryBuffer.h"

#include "cuda_runtime.h"
#include "Logger.h"

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

	gmb->p_rw = new unsigned char[(slots+1) *2 + meta_size];
	memset(gmb->p_rw, 0, (slots+1) * 2 + meta_size);
	
	gmb->h_mutex = CreateMutex(NULL, FALSE, NULL);
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

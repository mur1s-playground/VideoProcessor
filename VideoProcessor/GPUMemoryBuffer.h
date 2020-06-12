#pragma once

#include <windows.h>
#include <string>

using namespace std;

struct gpu_memory_buffer {
	string name;

	HANDLE h_mutex;
	unsigned char* p_device;

	unsigned char* p_rw;

	int size;
	int slots;
	int meta_size;

	bool error;
};

void gpu_memory_buffer_init(struct gpu_memory_buffer* gmb, const char* name, int size, int slots, int meta_size);

bool gpu_memory_buffer_try_rw(struct gpu_memory_buffer* gmb, int slot, bool block, int sleep_ms);
void gpu_memory_buffer_release_rw(struct gpu_memory_buffer* gmb, int slot);
bool gpu_memory_buffer_try_r(struct gpu_memory_buffer* gmb, int slot, bool block, int sleep_ms);
void gpu_memory_buffer_release_r(struct gpu_memory_buffer* gmb, int slot);

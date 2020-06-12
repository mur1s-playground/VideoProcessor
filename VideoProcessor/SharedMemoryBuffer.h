#pragma once

#include <windows.h>

#include <string>

using namespace std;

struct shared_memory_buffer {
	string name;

	HANDLE h_mutex;
	HANDLE h_map_file;
	LPCTSTR p_buffer;
	unsigned char* p_buf_c;

	int size;
	int slots;
	int meta_size;

	bool error;
};

void shared_memory_buffer_init_default(struct shared_memory_buffer* smb);
void shared_memory_buffer_init(struct shared_memory_buffer* smb, const char* name, int size, int slots, int meta_size);
void shared_memory_buffer_destroy(struct shared_memory_buffer* smb);

bool shared_memory_buffer_try_rw(struct shared_memory_buffer* smb, int slot, bool block, int sleep_ms);
void shared_memory_buffer_release_rw(struct shared_memory_buffer* smb, int slot);

bool shared_memory_buffer_try_r(struct shared_memory_buffer* smb, int slot, bool block, int sleep_ms);
void shared_memory_buffer_release_r(struct shared_memory_buffer* smb, int slot);


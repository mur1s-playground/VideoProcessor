#undef UNICODE

#include "SharedMemoryBuffer.h"

#include "Logger.h"

#include <sstream>
#include <fstream>

void shared_memory_buffer_init_default(struct shared_memory_buffer *smb) {
	shared_memory_buffer_init(smb, "default buffer", 1920*1080*4, 50, sizeof(int));
}

void shared_memory_buffer_init(struct shared_memory_buffer* smb, const char* name, int size, int slots, int meta_size) {
#undef UNICODE
	smb->error = false;
	
	smb->name = name;

	smb->size = size;
	smb->slots = slots;
	smb->meta_size = meta_size;

	int rw_locks_size = (slots+1) * 2;
	size_t size_complete = size*slots + rw_locks_size + meta_size;

	char* name_r = (char*)malloc(strlen(name) + 1);
	name_r[0] = 'c';
	strcpy_s(&name_r[1], strlen(name) + 1, name);

	smb->h_mutex = CreateMutex(NULL, FALSE, name_r);
	free(name_r);
	if (smb->h_mutex == NULL) {
		smb->error = true;
	}
	smb->h_map_file = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size_complete, name);
	if (smb->h_map_file == NULL) {
		smb->error = true;
	}
	smb->p_buffer = (LPTSTR)MapViewOfFile(smb->h_map_file, FILE_MAP_ALL_ACCESS, 0, 0, size_complete);
	smb->p_buf_c = (unsigned char*) smb->p_buffer;
	memset(smb->p_buf_c, 0, size_complete);
}

void shared_memory_buffer_edit(struct shared_memory_buffer* smb, const char* name, int size, int slots, int meta_size) {
	smb->error = false;

	smb->name = name;

	smb->size = size;
	smb->slots = slots;
	smb->meta_size = meta_size;

	int rw_locks_size = (slots + 1) * 2;
	size_t size_complete = size * slots + rw_locks_size + meta_size;

	char* name_r = (char*)malloc(strlen(name) + 2);
	name_r[0] = 'c';
	strcpy_s(&name_r[1], strlen(name) + 1, name);

	UnmapViewOfFile(smb->p_buffer);

	CloseHandle(smb->h_map_file);

	smb->h_map_file = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, size_complete, name);
	if (smb->h_map_file == NULL) {
		smb->error = true;
	}

	smb->p_buffer = (LPTSTR)MapViewOfFile(smb->h_map_file, FILE_MAP_ALL_ACCESS, 0, 0, size_complete);
	smb->p_buf_c = (unsigned char*)smb->p_buffer;
	memset(smb->p_buf_c, 0, size_complete);
}

bool shared_memory_buffer_try_rw(struct shared_memory_buffer* smb, int slot, bool block, int sleep_ms) {
	WaitForSingleObject(smb->h_mutex, INFINITE);
	bool is_written_to = (smb->p_buf_c[smb->size * smb->slots + 2 * slot + 1] == 1);
	bool is_read_from = (smb->p_buf_c[smb->size * smb->slots + 2 * slot] > 0);
	bool set_write = false;
	if (block) {
		while (true) {
			if (!is_written_to) {
				smb->p_buf_c[smb->size * smb->slots + 2 * slot + 1] = 1;
				set_write = true;
			}
			if (!is_read_from && set_write) {
				ReleaseMutex(smb->h_mutex);
				return true;
			}
			ReleaseMutex(smb->h_mutex);
			Sleep(sleep_ms);
			WaitForSingleObject(smb->h_mutex, INFINITE);
			is_written_to = (smb->p_buf_c[smb->size * smb->slots + 2 * slot + 1] == 1);
			is_read_from = (smb->p_buf_c[smb->size * smb->slots + 2 * slot] > 0);
		}
	} else {
		if (!is_written_to && !is_read_from) {
			smb->p_buf_c[smb->size * smb->slots + 2 * slot + 1] = 1;
			ReleaseMutex(smb->h_mutex);
			return true;
		}
	}
	ReleaseMutex(smb->h_mutex);
	return false;
}

void shared_memory_buffer_release_rw(struct shared_memory_buffer* smb, int slot) {
	WaitForSingleObject(smb->h_mutex, INFINITE);
	smb->p_buf_c[smb->size * smb->slots + 2 * slot + 1] = 0;
	ReleaseMutex(smb->h_mutex);
}

bool shared_memory_buffer_try_r(struct shared_memory_buffer* smb, int slot, bool block, int sleep_ms) {
	WaitForSingleObject(smb->h_mutex, INFINITE);
	bool is_written_to = (smb->p_buf_c[smb->size * smb->slots + 2 * slot + 1] == 1);
	if (block) {
		while (true) {
			if (!is_written_to) {
				smb->p_buf_c[smb->size * smb->slots + 2 * slot]++;
				ReleaseMutex(smb->h_mutex);
				return true;
			}
			ReleaseMutex(smb->h_mutex);
			Sleep(sleep_ms);
			WaitForSingleObject(smb->h_mutex, INFINITE);
			is_written_to = (smb->p_buf_c[smb->size * smb->slots + 2 * slot + 1] == 1);
		}
	} else {
		if (!is_written_to) {
			smb->p_buf_c[smb->size * smb->slots + 2 * slot]++;
			ReleaseMutex(smb->h_mutex);
			return true;
		}
	}
	ReleaseMutex(smb->h_mutex);
	return false;
}

void shared_memory_buffer_release_r(struct shared_memory_buffer* smb, int slot) {
	WaitForSingleObject(smb->h_mutex, INFINITE);
	smb->p_buf_c[smb->size * smb->slots + 2 * slot]--;
	ReleaseMutex(smb->h_mutex);
}
/*
struct application_graph_node {
	int n_id;

	void* component;
};
*/
void shared_memory_buffer_externalise(struct application_graph_node* agn, string& out_str) {
	/*
	struct shared_memory_buffer* smb = (struct shared_memory_buffer*)agn->component;

	stringstream s_out;
	s_out << smb->name << std::endl;
	s_out << smb->size << std::endl;
	s_out << smb->slots << std::endl;
	s_out << smb->meta_size << std::endl;

	out_str = s_out.str();
	*/
}

void shared_memory_buffer_load(struct shared_memory_buffer* smb, ifstream& in_f) {
	std::string line;
	std::getline(in_f, line);
	string name = line;
	std::getline(in_f, line);
	int size = stoi(line);
	std::getline(in_f, line);
	int slots = stoi(line);
	std::getline(in_f, line);
	int meta_size = stoi(line);

	shared_memory_buffer_init(smb, name.c_str(), size, slots, meta_size);
}

//TODO: complete
void shared_memory_buffer_destroy(struct application_graph_node* agn) {
	/*
	struct shared_memory_buffer* smb = (struct shared_memory_buffer*)agn->component;

	if (smb->h_mutex != NULL) {
		CloseHandle(smb->h_mutex);
	}
	if (smb->h_map_file != NULL) {
		CloseHandle(smb->h_map_file);
	}

	delete smb;
	*/
}
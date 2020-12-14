#pragma once

#include <string>
#include <windows.h>

using namespace std;

struct bit_field {
	unsigned int pages;
	unsigned int pagesize;
	unsigned int* data;

	unsigned char** invalidators;
	unsigned int invalidators_c;

	unsigned int** device_data;
	unsigned int* device_ids;
	unsigned int devices_c;

	unsigned int biggest_tracked_allocated_page;

	HANDLE* device_locks;
	//FIX: multigpu
	unsigned int device_add_pages;
};

void bit_field_init(struct bit_field* bf, unsigned int pages, unsigned int pagesize);

unsigned int bit_field_add_data(struct bit_field* bf, const unsigned int datum);
unsigned int bit_field_add_bulk(struct bit_field* bf, const unsigned int* data, const unsigned int data_len_in_bf, const unsigned int data_len_in_mem);
unsigned int bit_field_add_bulk_zero(struct bit_field* bf, const unsigned int data_len_in_bf);

unsigned int bit_field_add_data_to_segment(struct bit_field* bf, const unsigned int index, const unsigned int datum);
unsigned int bit_field_add_bulk_to_segment(struct bit_field* bf, const unsigned int index, const unsigned int* data, const unsigned int data_len_in_bf, const unsigned int data_len_in_mem);

void bit_field_update_data(struct bit_field* bf, const unsigned int index, const unsigned int datum);
void bit_field_update_bulk(struct bit_field* bf, const unsigned int index, const unsigned int* data, const unsigned int data_len_in_bf, const unsigned int data_len_in_mem);
void bit_field_invalidate_bulk(struct bit_field* bf, const unsigned int index, const unsigned int data_len_in_bf);

unsigned int bit_field_remove_data_from_segment(struct bit_field* bf, const unsigned int index, const unsigned int datum);
void bit_field_remove_bulk_from_segment(struct bit_field* bf, const unsigned int index);

unsigned int bit_field_register_device(struct bit_field* bf, unsigned int device_id);
void bit_field_update_device(struct bit_field* bf, unsigned int device_id);

void bit_field_update_host(struct bit_field* bf, unsigned int device_id);

void bit_field_load_from_disk(struct bit_field* bf, std::string filepath);
void bit_field_save_to_disk(const struct bit_field* bf, std::string filepath);

void bit_field_free(struct bit_field* bf);

void bit_field_dump(const struct bit_field* bf);
void bit_field_dump_invalidators(const struct bit_field* bf);
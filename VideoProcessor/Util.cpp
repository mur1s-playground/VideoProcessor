#include "Util.h"

#include "windows.h"
#include <fstream>
#include <sstream>

#include "Logger.h"

void util_read_binary(string filename, unsigned char *bin, size_t *out_length) {
	HANDLE file_handle = CreateFileA(filename.c_str(),
		FILE_GENERIC_READ,
		0,
		NULL,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL,
		NULL);

	(*out_length) = 0;

	if (file_handle != INVALID_HANDLE_VALUE) {
		char buffer[1024];
		memset(buffer, 0, 1024);

		DWORD dwBytesRead;

		unsigned char* bin_tmp = bin;
		
		while(ReadFile(file_handle, buffer, 1024, &dwBytesRead, NULL)) {
			if (dwBytesRead != 0) {
				memcpy(bin_tmp, buffer, dwBytesRead);
				bin_tmp += dwBytesRead;
				(*out_length) += dwBytesRead;
			} else {
				break;
			}
		}
	}

	CloseHandle(file_handle);
}

void util_write_binary(string filename, unsigned char *bin, size_t length) {
	HANDLE file_handle = CreateFileA(filename.c_str(),
		FILE_GENERIC_WRITE,
		0,
		NULL,
		OPEN_ALWAYS,
		FILE_ATTRIBUTE_NORMAL,
		NULL);

	char buffer[1024];
	memset(buffer, 0, 1024);

	DWORD dwBytesWritten;

	int ct = 0;
	while (ct < length) {
		int bytes_to_write = 1024;
		if (length - ct < 1024) {
			bytes_to_write = length - ct;
		}
		memcpy(buffer, &bin[ct], bytes_to_write);
		int part_write = 0;
		while (part_write < bytes_to_write) {
			WriteFile(file_handle, &buffer[part_write], bytes_to_write - part_write, &dwBytesWritten, NULL);
			part_write += dwBytesWritten;
			ct += dwBytesWritten;
		}
	}
	CloseHandle(file_handle);
}
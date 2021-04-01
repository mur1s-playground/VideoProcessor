#pragma once

#include <string>

using namespace std;

void util_read_binary(string filename, unsigned char* bin, size_t* out_length);
void util_write_binary(string filename, unsigned char* bin, size_t length);
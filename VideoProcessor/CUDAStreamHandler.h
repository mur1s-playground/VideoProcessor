#pragma once

#include "cuda_runtime.h"

void cuda_stream_handler_init();

extern cudaStream_t cuda_streams[5];


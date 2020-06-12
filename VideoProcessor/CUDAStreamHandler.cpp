#include "CUDAStreamHandler.h"

cudaStream_t cuda_streams[5];

void cuda_stream_handler_init() {
	for (int i = 0; i < 5; i++) {
		cudaStreamCreate(&cuda_streams[i]);
	}
}
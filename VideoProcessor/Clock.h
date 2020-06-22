#pragma once

#include <Windows.h>
#include <time.h>

#define BILLION                             (1E9)

static BOOL g_first_time = 1;
static LARGE_INTEGER g_counts_per_sec;

int clock_gettime(int dummy, struct timespec* ct);

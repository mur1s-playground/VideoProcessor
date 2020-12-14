#pragma once

#include <cuda_runtime.h>
#include <array>

template<typename T>
struct vector2 {
    T v[2];

    __host__ __device__ vector2() {}
    __host__ __device__ vector2(T x, T y) : v{ x, y } {}

    __host__ __device__ T& operator[](const std::size_t n) { return v[n]; }
    __host__ __device__ const T& operator[](const std::size_t n) const { return v[n]; }
};

template<typename T>
__host__ __device__ auto operator-(const vector2<T> v) -> vector2<T> {
    return { -v[0], -v[1] };
}

template<typename T>
__host__ __device__ auto operator-(const vector2<T> v1, const vector2<T> v2) -> vector2<T> {
    return { v1[0] - v2[0], v1[1] - v2[1] };
}
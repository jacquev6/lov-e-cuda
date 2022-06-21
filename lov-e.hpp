// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

// Documentation and license information for this file are available at https://github.com/jacquev6/lov-e-cuda

#ifndef LOV_E_HPP_
#define LOV_E_HPP_

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>


#ifdef __NVCC__
#define HOST_DEVICE_DECORATORS __host__ __device__
#else
#define HOST_DEVICE_DECORATORS
#endif


/*                          *
 * Error checking utilities *
 *                          */

struct CudaError : public std::exception {
  CudaError(const char* const file_, const unsigned line_, const cudaError_t error_) :
      file(file_), line(line_), error(error_) {
    snprintf(
      _what, sizeof(_what),
      "CUDA ERROR, detected at %s:%i: %i %s",
      file, line, static_cast<unsigned>(error), cudaGetErrorName(error));
  }

  const char* what() const noexcept override {
    return _what;
  }

  const char* const file;
  const unsigned line;
  const cudaError_t error;

 private:
  char _what[128];
};

// @todo Test the __device__ version
HOST_DEVICE_DECORATORS
inline void check_cuda_errors_no_sync_(const char* file, const unsigned line) {
  const cudaError_t error = cudaGetLastError();
  if (error) {
    #ifdef __CUDA_ARCH__
      printf(
        "CUDA ERROR, detected at %s:%i: %i %s\n",
        file, line, static_cast<unsigned int>(error), cudaGetErrorName(error));
    #else
      throw CudaError(file, line, error);
    #endif
  }
}

#define check_cuda_errors_no_sync() check_cuda_errors_no_sync_(__FILE__, __LINE__)

HOST_DEVICE_DECORATORS
inline void check_cuda_errors_(const char* const file, const unsigned line) {
  cudaDeviceSynchronize();
  check_cuda_errors_no_sync_(file, line);
}

#define check_cuda_errors() check_cuda_errors_(__FILE__, __LINE__)

/*                         *
 * Basic memory management *
 *                         */

template<typename T>
T* alloc_host(const std::size_t n) {
  if (n == 0) {
    return nullptr;
  } else {
    return reinterpret_cast<T*>(std::malloc(n * sizeof(T)));
  }
}

template<typename T>
HOST_DEVICE_DECORATORS
T* alloc_device(const std::size_t n) {
  if (n == 0) {
    return nullptr;
  } else {
    T* p;
    cudaMalloc(&p, n * sizeof(T));
    check_cuda_errors();
    return p;
  }
}

template<typename T>
void memset_host(const std::size_t n, const char v, T* const p) {
  std::memset(p, v, n * sizeof(T));
}

template<typename T>
void memset_device(const std::size_t n, const char v, T* const p) {
  cudaMemset(p, v, n * sizeof(T));
}

template<typename T>
void memreset_host(const std::size_t n, T* const p) {
  memset_host(n, 0, p);
}

template<typename T>
void memreset_device(const std::size_t n, T* const p) {
  memset_device(n, 0, p);
}

template<typename T>
void copy_host_to_device(const std::size_t n, const T* const src, T* const dst) {
  if (n == 0) {
    return;
  } else {
    cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice);
    check_cuda_errors();
  }
}

template<typename T>
T* clone_host_to_device(const std::size_t n, const T* const src) {
  T* dst = alloc_device<T>(n);
  copy_host_to_device(n, src, dst);
  return dst;
}

template<typename T>
void copy_device_to_host(const std::size_t n, const T* const src, T* const dst) {
  if (n == 0) {
    return;
  } else {
    cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost);
    check_cuda_errors();
  }
}

template<typename T>
T* clone_device_to_host(const std::size_t n, const T* const src) {
  T* dst = alloc_host<T>(n);
  copy_device_to_host(n, src, dst);
  return dst;
}

template<typename T>
void free_host(T* const p) {
  if (p == nullptr) {
    return;
  } else {
    std::free(p);
  }
}

template<typename T>
HOST_DEVICE_DECORATORS
void free_device(T* const p) {
  if (p == nullptr) {
    return;
  } else {
    cudaFree(p);
    check_cuda_errors();
  }
}

#endif  // LOV_E_HPP_

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


struct CudaError : public std::exception {
  CudaError(const char* file_, unsigned line_, cudaError_t error_) :
      file(file_), line(line_), error(error_) {
    snprintf(
      _what, sizeof(_what),
      "CUDA ERROR, detected at %s:%i: %i %s",
      file, line, static_cast<unsigned>(error), cudaGetErrorName(error));
  }

  const char* what() const noexcept override {
    return _what;
  }

  const char* file;
  unsigned line;
  cudaError_t error;

 private:
  char _what[128];
};

// @todo Test the __device__ version
HOST_DEVICE_DECORATORS
inline void check_cuda_errors_no_sync_(const char* file, unsigned line) {
  cudaError_t error = cudaGetLastError();
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
inline void check_cuda_errors_(const char* file, unsigned line) {
  cudaDeviceSynchronize();
  check_cuda_errors_no_sync_(file, line);
}

#define check_cuda_errors() check_cuda_errors_(__FILE__, __LINE__)

#endif  // LOV_E_HPP_

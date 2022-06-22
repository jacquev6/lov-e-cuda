// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

// Documentation and license information for this file are available at https://github.com/jacquev6/lov-e-cuda

#ifndef LOV_E_HPP_
#define LOV_E_HPP_

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <exception>


#ifdef __NVCC__
#define HOST_DEVICE_DECORATORS __host__ __device__
#define DEVICE_DECORATOR __device__
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

/*                           *
 * Generic memory management *
 *                           */

struct Host {
  template<typename T>
  static T* alloc(const std::size_t n) { return alloc_host<T>(n); }
  template<typename T>
  static void free(T* p) { return free_host(p); }
};

struct Device {
  template<typename T>
  static T* alloc(const std::size_t n) { return alloc_device<T>(n); }
  template<typename T>
  static void free(T* p) { return free_device(p); }
};

template<typename WhereFrom>
struct From {
  template<typename WhereTo>
  struct To;
};

template<>
template<>
struct From<Host>::To<Device> {
  template<typename T>
  static void copy(const std::size_t n, const T* const src, T* const dst) {
    return copy_host_to_device(n, src, dst);
  }

  template<typename T>
  static T* clone(const std::size_t n, const T* const src) {
    return clone_host_to_device(n, src);
  }
};

template<>
template<>
struct From<Device>::To<Host> {
  template<typename T>
  static void copy(const std::size_t n, const T* const src, T* const dst) {
    return copy_device_to_host(n, src, dst);
  }

  template<typename T>
  static T* clone(const std::size_t n, const T* const src) {
    return clone_device_to_host(n, src);
  }
};

/*            *
 * ArrayViews *
 *            */

template<typename Where, typename T>
class ArrayView1D;

// Provide specializations for each possible 'Where', to decorate (__host__ and/or __device__) the
// 'T& operator[](unsigned i0)' to avoid *during compilation* dereferencing a device pointer
// on the host and reciprocally.
// Downside: some code duplication, but I don't see a way to avoid it.
// @todo Maybe we can specialize only the operator[]? (https://stackoverflow.com/a/5950287/905845)
template<typename T>
class ArrayView1D<Host, T> {
 public:
  HOST_DEVICE_DECORATORS
  ArrayView1D(unsigned s0, T* data) : _s0(s0), _data(data) {}

  template<typename W, typename U>
  friend class ArrayView1D;

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D(const ArrayView1D<Host, U>& o) : _s0(o._s0), _data(o._data) {}

  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  inline T& operator[](unsigned i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

 protected:
  const unsigned _s0;
  T* const _data;
};

#ifdef __NVCC__

template<typename T>
class ArrayView1D<Device, T> {
 public:
  HOST_DEVICE_DECORATORS
  ArrayView1D(unsigned s0, T* data) : _s0(s0), _data(data) {}

  template<typename W, typename U>
  friend class ArrayView1D;

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D(const ArrayView1D<Device, U>& o) : _s0(o._s0), _data(o._data) {}

  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  DEVICE_DECORATOR
  inline T& operator[](unsigned i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

 protected:
  const unsigned _s0;
  T* const _data;
};

#endif

template<typename Where, typename T>
class ArrayView2D {
 public:
  HOST_DEVICE_DECORATORS
  ArrayView2D(unsigned s1, unsigned s0, T* data) : _s1(s1), _s0(s0), _data(data) {}

  template<typename W, typename U>
  friend class ArrayView2D;

  template<typename From, typename To, typename U>
  friend void copy(ArrayView2D<From, U> src, ArrayView2D<To, U> dst);

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView2D(const ArrayView2D<Where, U>& o) : _s1(o._s1), _s0(o._s0), _data(o._data) {}

  HOST_DEVICE_DECORATORS
  unsigned s1() const { return _s1; }
  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  inline ArrayView1D<Where, T> operator[](unsigned i1) const {
    assert(i1 < _s1);
    return ArrayView1D<Where, T>(_s0, _data + i1 * _s0);
  }

 protected:
  const unsigned _s1;
  const unsigned _s0;
  T* const _data;
};


template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView2D<WhereFrom, T> src, ArrayView2D<WhereTo, T> dst) {  // NOLINT(build/include_what_you_use)
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s1() * src.s0(), src._data, dst._data);
}

/*        *
 * Arrays *
 *        */

template<typename Where, typename T>
class Array1D : public ArrayView1D<Where, T> {
 public:
  explicit Array1D(unsigned s0) : ArrayView1D<Where, T>(s0, Where::template alloc<T>(s0)) {}
  ~Array1D() { Where::free(this->_data); }

//  private:
//   template<typename U>
//   Array1D(const Array1D<Where, U>&) = delete;
//   template<typename U>
//   Array1D(const Array1D<Where, U>&&) = delete;
};

template<typename Where, typename T>
class Array2D : public ArrayView2D<Where, T> {
 public:
  Array2D(unsigned s1, unsigned s0) : ArrayView2D<Where, T>(s1, s0, Where::template alloc<T>(s1 * s0)) {}
  ~Array2D() { Where::free(this->_data); }

//  private:
//   template<typename U>
//   Array2D(const Array2D<Where, U>&) = delete;
//   template<typename U>
//   Array2D(const Array2D<Where, U>&&) = delete;
};


struct Grid {
  const dim3 blocks;
  const dim3 threads;
};

#define CONFIG(grid) grid.blocks, grid.threads

/*
  Factories to make Grids with a given number of threads per block.
*/
template<unsigned BLOCKDIM_X, unsigned BLOCKDIM_Y>
struct GridFactory2D {
  HOST_DEVICE_DECORATORS
  static Grid make(int x, int y) {
    return Grid {
      dim3(
        (x + BLOCKDIM_X - 1) / BLOCKDIM_X,
        (y + BLOCKDIM_Y - 1) / BLOCKDIM_Y,
        1),
      dim3(BLOCKDIM_X, BLOCKDIM_Y, 1),
    };
  }

  HOST_DEVICE_DECORATORS
  static Grid fixed(int x, int y) {
    return Grid {
      dim3(x, y, 1),
      dim3(BLOCKDIM_X, BLOCKDIM_Y, 1),
    };
  }
};

/*
*/

#ifdef __NVCC__

/*
  Structures to factorize the pervasive 'blockIdx.x * blockDim.x + threadIdx.x'.
  Just replace that with 'Block?D<...>::x()'.

  *Note* that if you first:
      typedef GridFactory2D<BLOCKDIM_X, BLOCKDIM_Y> grid;
      typedef Block2D<BLOCKDIM_X, BLOCKDIM_Y> block;
  then the syntax is much lighter:
  In the kernel:
      const unsigned x = block::x();
      const unsigned y = block::y();
  At call site:
      auto grid = grid::make(width, height);
      kernel<<<CONFIG(grid)>>>();
*/
template<unsigned BLOCKDIM_X, unsigned BLOCKDIM_Y>
struct Block2D {
  __device__ static unsigned x() {
    assert(blockDim.x == BLOCKDIM_X);
    return blockIdx.x * BLOCKDIM_X + threadIdx.x;
  }

  __device__ static unsigned y() {
    assert(blockDim.y == BLOCKDIM_Y);
    return blockIdx.y * BLOCKDIM_Y + threadIdx.y;
  }
};

#endif  // __NVCC__

#undef HOST_DEVICE_DECORATORS
#undef DEVICE_DECORATOR

#endif  // LOV_E_HPP_

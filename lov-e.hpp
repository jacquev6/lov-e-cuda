// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

// Documentation and license information for this file are available at https://github.com/jacquev6/lov-e-cuda

#ifndef LOV_E_HPP_
#define LOV_E_HPP_

#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <exception>
#include <type_traits>


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
  // @todo Double-check that all error conditions can be detected using `cudaGetLastError`
  // (i.e. that no error condition can *only* be detected in the return value of the function)
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

// @todo Add parameter [uninitialized|zeroed] to `alloc_*` and Array?D
// @todo Statically ensure that T is trivially constructible and copyable
// (its Ctor will not be called when we use malloc/free)
// @todo Consider initializing the memory to a 'weird' value in debug mode.
// Maybe specialize that value by type? Signaling NaN has some appeal for floats, what about ints?

/*                   *
 * Memory management *
 *                   */

struct Host {
  template<typename T>
  static void memset(const std::size_t n, const char v, T* const p) {
    std::memset(p, v, n * sizeof(T));
  }

  template<typename T>
  static void memreset(const std::size_t n, T* const p) {
    memset(n, 0, p);
  }

  template<typename T>
  static T* alloc(const std::size_t n) {
    static_assert(std::is_trivial_v<T>);
    if (n == 0) {
      return nullptr;
    } else {
      return reinterpret_cast<T*>(std::malloc(n * sizeof(T)));
    }
  }

  template<typename T>
  static void free(T* p) {
    if (p == nullptr) {
      return;
    } else {
      std::free(p);
    }
  }
};

struct Device {
  template<typename T>
  static void memset(const std::size_t n, const char v, T* const p) {
    cudaMemset(p, v, n * sizeof(T));
  }

  template<typename T>
  static void memreset(const std::size_t n, T* const p) {
    memset(n, 0, p);
  }

  template<typename T>
  HOST_DEVICE_DECORATORS
  static T* alloc(const std::size_t n) {
    static_assert(std::is_trivial_v<T>);
    // @todoc If your type is not technically trivial but you still think it's safe to 'malloc',
    // you have two choices: 1) make it technically trivial or 2) implement your own versions of
    // Hst and Device and use them instead of the ones provided by *Lov-e-cuda*.
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
  HOST_DEVICE_DECORATORS
  static void free(T* p) {
    if (p == nullptr) {
      return;
    } else {
      cudaFree(p);
      check_cuda_errors();
    }
  }
};

template<typename WhereFrom>
struct From {
  template<typename WhereTo>
  struct To {
    template<typename T>
    static void copy(const std::size_t n, const T* const src, T* const dst);

    template<typename T>
    static T* clone(const std::size_t n, const T* const src) {
      T* dst = WhereTo::template alloc<T>(n);
      copy(n, src, dst);
      return dst;
    }
  };
};

template<> template<> template<typename T>
void From<Host>::To<Host>::copy(const std::size_t n, const T* const src, T* const dst) {
  if (n == 0) {
    return;
  } else {
    std::memcpy(dst, src, n * sizeof(T));
  }
}

template<> template<> template<typename T>
void From<Host>::To<Device>::copy(const std::size_t n, const T* const src, T* const dst) {
  if (n == 0) {
    return;
  } else {
    cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice);
    check_cuda_errors();
  }
}

template<> template<> template<typename T>
void From<Device>::To<Device>::copy(const std::size_t n, const T* const src, T* const dst) {
  if (n == 0) {
    return;
  } else {
    cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice);
    check_cuda_errors();
  }
}

template<> template<> template<typename T>
void From<Device>::To<Host>::copy(const std::size_t n, const T* const src, T* const dst) {
  if (n == 0) {
    return;
  } else {
    cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost);
    check_cuda_errors();
  }
}

/*            *
 * ArrayViews *
 *            */

// These classes have "non-owning pointer" semantics, so they follow the
// [Rule of Zero](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-zero)

template<typename Where, typename T>
class ArrayView1D;

// Provide specializations for each possible 'Where', to decorate (__host__ and/or __device__) the
// 'T& operator[](unsigned i0)' to avoid *during compilation* dereferencing a device pointer
// on the host and reciprocally.
// Downside: some code duplication, but I don't see a way to avoid it.
template<typename T>
class ArrayView1D<Host, T> {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView1D(unsigned s0, T* data) : _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D(const ArrayView1D<Host, U>& o) : _s0(o.s0()), _data(o.data_for_legacy_use()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D& operator=(const ArrayView1D<Host, U>& o) {
    _s0 = o.s0();
    _data = o.data_for_legacy_use();
    return *this;
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  T& operator[](unsigned i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

  HOST_DEVICE_DECORATORS
  T* data_for_legacy_use() const { return _data; }

 private:
  unsigned _s0;
  T* _data;
};

#ifdef __NVCC__

template<typename T>
class ArrayView1D<Device, T> {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView1D(unsigned s0, T* data) : _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D(const ArrayView1D<Device, U>& o) : _s0(o.s0()), _data(o.data_for_legacy_use()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D& operator=(const ArrayView1D<Device, U>& o) {
    _s0 = o.s0();
    _data = o.data_for_legacy_use();
    return *this;
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  DEVICE_DECORATOR
  T& operator[](unsigned i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

  HOST_DEVICE_DECORATORS
  T* data_for_legacy_use() const { return _data; }

 private:
  unsigned _s0;
  T* _data;
};

#endif

template<typename Where, typename T>
class ArrayView2D {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView2D(unsigned s1, unsigned s0, T* data) : _s1(s1), _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView2D(const ArrayView2D<Where, U>& o) : _s1(o.s1()), _s0(o.s0()), _data(o.data_for_legacy_use()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView2D& operator=(const ArrayView2D<Where, U>& o) {
    _s1 = o.s1();
    _s0 = o.s0();
    _data = o.data_for_legacy_use();
    return *this;
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  unsigned s1() const { return _s1; }

  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  ArrayView1D<Where, T> operator[](unsigned i1) const {
    assert(i1 < _s1);
    return ArrayView1D<Where, T>(_s0, _data + i1 * _s0);
  }

  HOST_DEVICE_DECORATORS
  T* data_for_legacy_use() const { return _data; }

 private:
  unsigned _s1;
  unsigned _s0;
  T* _data;
};

template<typename Where, typename T>
class ArrayView3D {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView3D(unsigned s2, unsigned s1, unsigned s0, T* data) : _s2(s2), _s1(s1), _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView3D(const ArrayView3D<Where, U>& o) : _s2(o.s2()), _s1(o.s1()), _s0(o.s0()), _data(o.data_for_legacy_use()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView3D& operator=(const ArrayView3D<Where, U>& o) {
    _s2 = o.s2();
    _s1 = o.s1();
    _s0 = o.s0();
    _data = o.data_for_legacy_use();
    return *this;
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  unsigned s2() const { return _s2; }

  HOST_DEVICE_DECORATORS
  unsigned s1() const { return _s1; }

  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  ArrayView2D<Where, T> operator[](unsigned i2) const {
    assert(i2 < _s2);
    return ArrayView2D<Where, T>(_s1, _s0, _data + i2 * _s1);
  }

  HOST_DEVICE_DECORATORS
  T* data_for_legacy_use() const { return _data; }

 private:
  unsigned _s2;
  unsigned _s1;
  unsigned _s0;
  T* _data;
};

// @todo Support dimensions up to 5
// @todo Evaluate if we could use a template definition where the dimension is a template parameter
// and use typedefs for dimensions 1 to 5.
// @todo Use typedefs for Host/Device as well.

// @todo Add `copy` for other dimensions
template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView1D<WhereFrom, T> src, ArrayView1D<WhereTo, T> dst) {
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s0(), src.data_for_legacy_use(), dst.data_for_legacy_use());
}

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView2D<WhereFrom, T> src, ArrayView2D<WhereTo, T> dst) {
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s1() * src.s0(), src.data_for_legacy_use(), dst.data_for_legacy_use());
}

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView3D<WhereFrom, T> src, ArrayView3D<WhereTo, T> dst) {
  assert(dst.s2() == src.s2());
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s2() * src.s1() * src.s0(), src.data_for_legacy_use(), dst.data_for_legacy_use());
}

/*        *
 * Arrays *
 *        */

// These classes have "owning pointer" semantics, so they follow the
// [Rule of Five](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-five)

// @todo Is inheritance the best here?
template<typename Where, typename T>
class Array1D : public ArrayView1D<Where, T> {
 public:
  explicit Array1D(unsigned s0) : ArrayView1D<Where, T>(s0, Where::template alloc<T>(s0)) {}
  ~Array1D() { Where::free(this->data_for_legacy_use()); }

  // @todo Make thorough inventory of compiler-generated functions (operator=, constructors, etc.)
  // and implement/delete them as required
};

template<typename Where, typename T>
class Array2D : public ArrayView2D<Where, T> {
 public:
  Array2D(unsigned s1, unsigned s0) : ArrayView2D<Where, T>(s1, s0, Where::template alloc<T>(s1 * s0)) {}
  ~Array2D() { Where::free(this->data_for_legacy_use()); }
};

// @todo Support dimensions up to 5

template<typename WhereTo, typename WhereFrom, typename T>
Array2D<WhereTo, T> clone_to(ArrayView2D<WhereFrom, T> src) {
  Array2D<WhereTo, T> dst(src.s1(), src.s0());
  copy(src, dst);  // NOLINT(build/include_what_you_use)
  return dst;  // @todo Make it work even without RVO
  // (I think RVO is saving us from double-free until we implement a move constructor)
}
// @todo Add `clone_to` for other dimensions


/*                          *
 * Grid and block utilities *
 *                          */

struct Grid {
  const dim3 blocks;
  const dim3 threads;
};

#define CONFIG(grid) grid.blocks, grid.threads

// @todo Merge GridFactory?D and Block?D
// they should always be used with the same BLOCKDIM_X, BLOCKDIM_Y, so keeping them separate is error-prone
// @todo Consider giving a similar way to create Grids but with block sizes decided at runtime
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


#ifdef __NVCC__

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

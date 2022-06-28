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
      "CUDA ERROR, detected at %s:%i: code %i=%s: %s",
      file, line, static_cast<unsigned>(error), cudaGetErrorName(error), cudaGetErrorString(error));
  }

  const char* what() const noexcept override {
    return _what;
  }

  const char* const file;
  const unsigned line;
  const cudaError_t error;

 private:
  char _what[256];
};

// @todo Test the __device__ versions of the error checking functions

HOST_DEVICE_DECORATORS
inline void check_cuda_error_(const cudaError_t error, const char* file, const unsigned line) {
  if (error != cudaSuccess) {
    #ifdef __CUDA_ARCH__
      printf(
        "CUDA ERROR, detected at %s:%i: code %i=%s: %s\n",
        file, line, static_cast<unsigned int>(error), cudaGetErrorName(error), cudaGetErrorString(error));
    #else
      throw CudaError(file, line, error);
    #endif
  }
}

#define check_cuda_error(e) check_cuda_error_(e, __FILE__, __LINE__)

HOST_DEVICE_DECORATORS
inline void check_last_cuda_error_no_sync_(const char* file, const unsigned line) {
  check_cuda_error_(cudaGetLastError(), file, line);
}

#define check_last_cuda_error_no_sync() check_last_cuda_error_no_sync_(__FILE__, __LINE__)

HOST_DEVICE_DECORATORS
inline void check_last_cuda_error_(const char* const file, const unsigned line) {
  cudaDeviceSynchronize();
  check_last_cuda_error_no_sync_(file, line);
}

#define check_last_cuda_error() check_last_cuda_error_(__FILE__, __LINE__)

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
      T* const p = reinterpret_cast<T*>(std::malloc(n * sizeof(T)));
      #ifndef NDEBUG
        // Attempt to make use of uninitialized memory more noticeable by actually
        // initializing it to a large weird-looking value made of a repeated byte.
        // The repetition of the byte 0x66 yields the following values:
        // - 8-bits integer: 102
        // - 16-bits integer: 26214
        // - 32-bits integer: 1717986918
        // - 64-bits integer: 7378697629483820646
        // - IEEE 754 float: 2.72008e+23
        // - IEEE 754 double: 1.9035985662552932e+185
        memset(n, 0x66, p);
      #endif
      return p;
    }
  }

  template<typename T>
  static T* alloc_zeored(const std::size_t n) {
    T* const p = alloc<T>(n);
    memreset(n, p);
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
    check_cuda_error(cudaMemset(p, v, n * sizeof(T)));
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
      check_cuda_error(cudaMalloc(&p, n * sizeof(T)));
      return p;
    }
  }

  template<typename T>
  static T* alloc_zeored(const std::size_t n) {
    T* const p = alloc<T>(n);
    memreset(n, p);
  }

  template<typename T>
  HOST_DEVICE_DECORATORS
  static void free(T* p) {
    if (p == nullptr) {
      return;
    } else {
      check_cuda_error(cudaFree(p));
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
    check_cuda_error(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
}

template<> template<> template<typename T>
void From<Device>::To<Device>::copy(const std::size_t n, const T* const src, T* const dst) {
  if (n == 0) {
    return;
  } else {
    check_cuda_error(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
  }
}

template<> template<> template<typename T>
void From<Device>::To<Host>::copy(const std::size_t n, const T* const src, T* const dst) {
  if (n == 0) {
    return;
  } else {
    check_cuda_error(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost));
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

template<typename Where, typename T>
class ArrayView4D {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView4D(unsigned s3, unsigned s2, unsigned s1, unsigned s0, T* data) :
    _s3(s3), _s2(s2), _s1(s1), _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView4D(const ArrayView4D<Where, U>& o) :
    _s3(o.s3()), _s2(o.s2()), _s1(o.s1()), _s0(o.s0()), _data(o.data_for_legacy_use()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView4D& operator=(const ArrayView4D<Where, U>& o) {
    _s3 = o.s3();
    _s2 = o.s2();
    _s1 = o.s1();
    _s0 = o.s0();
    _data = o.data_for_legacy_use();
    return *this;
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  unsigned s3() const { return _s3; }

  HOST_DEVICE_DECORATORS
  unsigned s2() const { return _s2; }

  HOST_DEVICE_DECORATORS
  unsigned s1() const { return _s1; }

  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  ArrayView3D<Where, T> operator[](unsigned i3) const {
    assert(i3 < _s3);
    return ArrayView3D<Where, T>(_s2, _s1, _s0, _data + i3 * _s2);
  }

  HOST_DEVICE_DECORATORS
  T* data_for_legacy_use() const { return _data; }

 private:
  unsigned _s3;
  unsigned _s2;
  unsigned _s1;
  unsigned _s0;
  T* _data;
};

template<typename Where, typename T>
class ArrayView5D {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView5D(unsigned s4, unsigned s3, unsigned s2, unsigned s1, unsigned s0, T* data) :
    _s4(s4), _s3(s3), _s2(s2), _s1(s1), _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView5D(const ArrayView5D<Where, U>& o) :
    _s4(o.s4()), _s3(o.s3()), _s2(o.s2()), _s1(o.s1()), _s0(o.s0()), _data(o.data_for_legacy_use()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView5D& operator=(const ArrayView5D<Where, U>& o) {
    _s4 = o.s4();
    _s3 = o.s3();
    _s2 = o.s2();
    _s1 = o.s1();
    _s0 = o.s0();
    _data = o.data_for_legacy_use();
    return *this;
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  unsigned s4() const { return _s4; }

  HOST_DEVICE_DECORATORS
  unsigned s3() const { return _s3; }

  HOST_DEVICE_DECORATORS
  unsigned s2() const { return _s2; }

  HOST_DEVICE_DECORATORS
  unsigned s1() const { return _s1; }

  HOST_DEVICE_DECORATORS
  unsigned s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  ArrayView4D<Where, T> operator[](unsigned i4) const {
    assert(i4 < _s4);
    return ArrayView4D<Where, T>(_s3, _s2, _s1, _s0, _data + i4 * _s3);
  }

  HOST_DEVICE_DECORATORS
  T* data_for_legacy_use() const { return _data; }

 private:
  unsigned _s4;
  unsigned _s3;
  unsigned _s2;
  unsigned _s1;
  unsigned _s0;
  T* _data;
};

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

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView4D<WhereFrom, T> src, ArrayView4D<WhereTo, T> dst) {
  assert(dst.s3() == src.s3());
  assert(dst.s2() == src.s2());
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s3() * src.s2() * src.s1() * src.s0(), src.data_for_legacy_use(), dst.data_for_legacy_use());
}

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView5D<WhereFrom, T> src, ArrayView5D<WhereTo, T> dst) {
  assert(dst.s4() == src.s4());
  assert(dst.s3() == src.s3());
  assert(dst.s2() == src.s2());
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s4() *  src.s3() * src.s2() * src.s1() * src.s0(), src.data_for_legacy_use(), dst.data_for_legacy_use());
}

/*        *
 * Arrays *
 *        */

// Constructor parameters
enum Zeroed {zeroed};
enum Uninitialized {uninitialized};

// These classes have "owning pointer" semantics, so they follow the
// [Rule of Five](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-five)

template<typename Where, typename T>
class Array1D : public ArrayView1D<Where, T> {
 public:
  explicit Array1D(unsigned s0, Uninitialized) : ArrayView1D<Where, T>(s0, Where::template alloc<T>(s0)) {}
  explicit Array1D(unsigned s0, Zeroed) : ArrayView1D<Where, T>(s0, Where::template alloc_zeroed<T>(s0)) {}
  ~Array1D() { Where::free(this->data_for_legacy_use()); }

  // @todo Make thorough inventory of compiler-generated functions (operator=, constructors, etc.)
  // and implement/delete them as required
};

template<typename Where, typename T>
class Array2D : public ArrayView2D<Where, T> {
 public:
  Array2D(unsigned s1, unsigned s0, Uninitialized) : ArrayView2D<Where, T>(s1, s0, Where::template alloc<T>(s1 * s0)) {}
  Array2D(unsigned s1, unsigned s0, Zeroed) : ArrayView2D<Where, T>(s1, s0, Where::template alloc_zeroed<T>(s1 * s0)) {}
  ~Array2D() { Where::free(this->data_for_legacy_use()); }
};

// @todo Support dimensions up to 5

template<typename WhereTo, typename WhereFrom, typename T>
Array2D<WhereTo, T> clone_to(ArrayView2D<WhereFrom, T> src) {
  Array2D<WhereTo, T> dst(src.s1(), src.s0(), uninitialized);
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

#ifdef __NVCC__
  __device__ static unsigned x() {
    assert(blockDim.x == BLOCKDIM_X);
    return blockIdx.x * BLOCKDIM_X + threadIdx.x;
  }

  __device__ static unsigned y() {
    assert(blockDim.y == BLOCKDIM_Y);
    return blockIdx.y * BLOCKDIM_Y + threadIdx.y;
  }
#endif  // __NVCC__
};

#undef HOST_DEVICE_DECORATORS
#undef DEVICE_DECORATOR

#endif  // LOV_E_HPP_

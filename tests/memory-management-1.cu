// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


struct Trivial {
  int i;
  char c;
  bool b;
  float f;
  double d;
};

struct NonTrivial1 {
  NonTrivial1() {}
};

struct NonTrivial2 {
  virtual void f() {}
};

TEST(HostAllocTest, AllocateNonZero) {
  int* const p = Host::alloc<int>(10);
  EXPECT_NE(p, nullptr);
  Host::free(p);

  // Can allocate any trivial type (https://en.cppreference.com/w/cpp/named_req/TrivialType)
  Host::free(Host::alloc<float>(1));
  Host::free(Host::alloc<double>(1));
  Host::free(Host::alloc<Trivial>(1));

  // Can't allocate non-trivial types
  #if EXPECT_COMPILE_ERROR == __LINE__
    Host::alloc<NonTrivial1>(1);
  #endif
  #if EXPECT_COMPILE_ERROR == __LINE__
    Host::alloc<NonTrivial2>(1);
  #endif
}

TEST(HostAllocTest, AllocateZero) {
  int* const p = Host::alloc<int>(0);
  EXPECT_EQ(p, nullptr);
  Host::free(p);
}

class HostMemsetTest : public testing::Test {
 protected:
  void SetUp() override {
    mem = Host::alloc<uint16_t>(count);
  }

  void TearDown() override {
    Host::free(mem);
  }

  uint16_t* mem;
  const unsigned count = 16;
};

TEST_F(HostMemsetTest, Memset) {
  Host::memset(count, 0xAA, mem);
  EXPECT_EQ(mem[0], 0xAAAA);
  EXPECT_EQ(mem[count - 1], 0xAAAA);
}

TEST_F(HostMemsetTest, Memreset) {
  mem[0] = 0xABCD;
  mem[count - 1] = 0x1234;
  Host::memreset(count, mem);
  EXPECT_EQ(mem[0], 0);
  EXPECT_EQ(mem[count - 1], 0);
}

TEST(DeviceAllocTest, AllocateNonZeroOnHost) {
  int* const p = Device::alloc<int>(10);
  EXPECT_NE(p, nullptr);
  Device::free(p);

  // Can allocate any trivial type
  Device::free(Device::alloc<float>(1));
  Device::free(Device::alloc<double>(1));
  Device::free(Device::alloc<Trivial>(1));

  // Can't allocate non-trivial types
  #if EXPECT_COMPILE_ERROR == __LINE__
    Device::alloc<NonTrivial1>(1);
  #endif
  #if EXPECT_COMPILE_ERROR == __LINE__
    Device::alloc<NonTrivial2>(1);
  #endif
}

TEST(DeviceAllocTest, AllocateZeroOnHost) {
  int* const p = Device::alloc<int>(0);
  EXPECT_EQ(p, nullptr);
  Device::free(p);
}

class DeviceMemsetTest : public testing::Test {
 protected:
  void SetUp() override {
    mem = Device::alloc<uint16_t>(count);
  }

  void TearDown() override {
    Device::free(mem);
  }

  uint16_t* mem;
  const std::size_t count = 16;
};

__global__ void Devicekernel_MemsetTest_MemsetOnHost(const std::size_t count, const uint16_t* const mem) {
  assert(mem[0] == 0xAAAA);
  assert(mem[count - 1] == 0xAAAA);
}

TEST_F(DeviceMemsetTest, MemsetOnHost) {
  Device::memset(count, 0xAA, mem);
  Devicekernel_MemsetTest_MemsetOnHost<<<1, 1>>>(count, mem);
  check_cuda_errors();
}

__global__ void Devicekernel_MemsetTest_MemresetOnHost_1(const std::size_t count, uint16_t* const mem) {
  mem[0] = 0xABCD;
  mem[count - 1] = 0x1234;
}

__global__ void Devicekernel_MemsetTest_MemresetOnHost_2(const std::size_t count, const uint16_t* const mem) {
  assert(mem[0] == 0);
  assert(mem[count - 1] == 0);
}

TEST_F(DeviceMemsetTest, MemresetOnHost) {
  Devicekernel_MemsetTest_MemresetOnHost_1<<<1, 1>>>(count, mem);
  Device::memreset(count, mem);
  Devicekernel_MemsetTest_MemresetOnHost_2<<<1, 1>>>(count, mem);
  check_cuda_errors();
}

class CopyTest : public testing::Test {
 protected:
  void SetUp() override {
    h = Host::alloc<uint16_t>(count);
    d = Device::alloc<uint16_t>(count);
  }

  void TearDown() override {
    Host::free(h);
    Device::free(d);
  }

  uint16_t* h;
  uint16_t* d;
  const std::size_t count = 16;
};

__global__ void kernel_CopyTest_CopyHostToDevice(const std::size_t count, const uint16_t* const d) {
  assert(d[0] == 42);
  assert(d[count - 1] == 65);
}

TEST_F(CopyTest, CopyHostToDevice) {
  h[0] = 42;
  h[count - 1] = 65;
  From<Host>::To<Device>::copy(count, h, d);
  kernel_CopyTest_CopyHostToDevice<<<1, 1>>>(count, d);
  check_cuda_errors();
}

__global__ void kernel_CopyTest_CopyDeviceToHost(const std::size_t count, uint16_t* const d) {
  d[0] = 42;
  d[count - 1] = 65;
}

TEST_F(CopyTest, CopyDeviceToHost) {
  kernel_CopyTest_CopyDeviceToHost<<<1, 1>>>(count, d);
  From<Device>::To<Host>::copy(count, d, h);
  EXPECT_EQ(h[0], 42);
  EXPECT_EQ(h[count - 1], 65);
}

TEST(CloneTest, CloneHostToDevice) {
  const std::size_t count = 16;
  uint16_t* h = Host::alloc<uint16_t>(count);
  h[0] = 42;
  h[count - 1] = 65;
  uint16_t* d = From<Host>::To<Device>::clone(count, h);
  kernel_CopyTest_CopyHostToDevice<<<1, 1>>>(count, d);
  check_cuda_errors();

  Device::free(d);
  Host::free(h);
}

TEST(CloneTest, CloneDeviceToHost) {
  const std::size_t count = 16;
  uint16_t* d = Device::alloc<uint16_t>(count);
  kernel_CopyTest_CopyDeviceToHost<<<1, 1>>>(count, d);
  uint16_t* h = From<Device>::To<Host>::clone(count, d);
  EXPECT_EQ(h[0], 42);
  EXPECT_EQ(h[count - 1], 65);

  Host::free(h);
  printf("B: d=%p\n", d);
  Device::free(d);
}

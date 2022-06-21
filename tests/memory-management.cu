// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


TEST(AllocHostTest, AllocateNonZero) {
  int* const p = alloc_host<int>(10);
  EXPECT_NE(p, nullptr);
  free_host(p);
}

TEST(AllocHostTest, AllocateZero) {
  int* const p = alloc_host<int>(0);
  EXPECT_EQ(p, nullptr);
  free_host(p);
}

class MemsetHostTest : public testing::Test {
 protected:
  void SetUp() override {
    mem = alloc_host<uint16_t>(count);
  }

  void TearDown() override {
    free_host(mem);
  }

  uint16_t* mem;
  const unsigned count = 16;
};

TEST_F(MemsetHostTest, Memset) {
  memset_host(count, 0xAA, mem);
  EXPECT_EQ(mem[0], 0xAAAA);
  EXPECT_EQ(mem[count - 1], 0xAAAA);
}

TEST_F(MemsetHostTest, Memreset) {
  mem[0] = 0xABCD;
  mem[count - 1] = 0x1234;
  memreset_host(count, mem);
  EXPECT_EQ(mem[0], 0);
  EXPECT_EQ(mem[count - 1], 0);
}

TEST(AllocDeviceTest, AllocateNonZeroOnHost) {
  int* const p = alloc_device<int>(10);
  EXPECT_NE(p, nullptr);
  free_device(p);
}

TEST(AllocDeviceTest, AllocateZeroOnHost) {
  int* const p = alloc_device<int>(0);
  EXPECT_EQ(p, nullptr);
  free_device(p);
}

__global__ void kernel_AllocDevice_AllocateNonZeroOnDevice() {
  int* const p = alloc_device<int>(10);
  assert(p != nullptr);
  free_device(p);
}

TEST(AllocDeviceTest, AllocateNonZeroOnDevice) {
  kernel_AllocDevice_AllocateNonZeroOnDevice<<<1, 1>>>();
  check_cuda_errors();
}

__global__ void kernel_AllocDevice_AllocateZeroOnDevice() {
  int* const p = alloc_device<int>(0);
  assert(p == nullptr);
  free_device(p);
}

TEST(AllocDeviceTest, AllocateZeroOnDevice) {
  kernel_AllocDevice_AllocateZeroOnDevice<<<1, 1>>>();
  check_cuda_errors();
}

class MemsetDeviceTest : public testing::Test {
 protected:
  void SetUp() override {
    mem = alloc_device<uint16_t>(count);
  }

  void TearDown() override {
    free_device(mem);
  }

  uint16_t* mem;
  const std::size_t count = 16;
};

__global__ void kernel_MemsetDeviceTest_MemsetOnHost(const std::size_t count, const uint16_t* const mem) {
  assert(mem[0] == 0xAAAA);
  assert(mem[count - 1] == 0xAAAA);
}

TEST_F(MemsetDeviceTest, MemsetOnHost) {
  memset_device(count, 0xAA, mem);
  kernel_MemsetDeviceTest_MemsetOnHost<<<1, 1>>>(count, mem);
  check_cuda_errors();
}

__global__ void kernel_MemsetDeviceTest_MemresetOnHost_1(const std::size_t count, uint16_t* const mem) {
  mem[0] = 0xABCD;
  mem[count - 1] = 0x1234;
}

__global__ void kernel_MemsetDeviceTest_MemresetOnHost_2(const std::size_t count, const uint16_t* const mem) {
  assert(mem[0] == 0);
  assert(mem[count - 1] == 0);
}

TEST_F(MemsetDeviceTest, MemresetOnHost) {
  kernel_MemsetDeviceTest_MemresetOnHost_1<<<1, 1>>>(count, mem);
  memreset_device(count, mem);
  kernel_MemsetDeviceTest_MemresetOnHost_2<<<1, 1>>>(count, mem);
  check_cuda_errors();
}

class CopyTest : public testing::Test {
 protected:
  void SetUp() override {
    h = alloc_host<uint16_t>(count);
    d = alloc_device<uint16_t>(count);
  }

  void TearDown() override {
    free_host(h);
    free_device(d);
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
  copy_host_to_device(count, h, d);
  kernel_CopyTest_CopyHostToDevice<<<1, 1>>>(count, d);
  check_cuda_errors();
}

__global__ void kernel_CopyTest_CopyDeviceToHost(const std::size_t count, uint16_t* const d) {
  d[0] = 42;
  d[count - 1] = 65;
}

TEST_F(CopyTest, CopyDeviceToHost) {
  kernel_CopyTest_CopyDeviceToHost<<<1, 1>>>(count, d);
  copy_device_to_host(count, d, h);
  EXPECT_EQ(h[0], 42);
  EXPECT_EQ(h[count - 1], 65);
}

TEST(CloneTest, CloneHostToDevice) {
  const std::size_t count = 16;
  uint16_t* h = alloc_host<uint16_t>(count);
  h[0] = 42;
  h[count - 1] = 65;
  uint16_t* d = clone_host_to_device(count, h);
  kernel_CopyTest_CopyHostToDevice<<<1, 1>>>(count, d);
  check_cuda_errors();

  free_device(d);
  free_host(h);
}

TEST(CloneTest, CloneDeviceToHost) {
  const std::size_t count = 16;
  uint16_t* d = alloc_device<uint16_t>(count);
  kernel_CopyTest_CopyDeviceToHost<<<1, 1>>>(count, d);
  uint16_t* h = clone_device_to_host(count, d);
  EXPECT_EQ(h[0], 42);
  EXPECT_EQ(h[count - 1], 65);

  free_host(h);
  free_device(d);
}

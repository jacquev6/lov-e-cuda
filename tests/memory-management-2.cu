// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


// If this test is added to 'memory-management-1.cu' (where it belongs),
// then `cuda-memcheck` does not detect leaks. So we keep it in a separate file.
__global__ void kernel_AllocDevice_AllocateNonZeroOnDevice() {
  int* const p = Device::alloc<int>(10);
  printf("A: p=%p\n", p);
  assert(p != nullptr);
  Device::free(p);
}

TEST(DeviceAllocTest, AllocateNonZeroOnDevice) {
  kernel_AllocDevice_AllocateNonZeroOnDevice<<<1, 1>>>();
  check_cuda_errors();
}

__global__ void kernel_AllocDevice_AllocateZeroOnDevice() {
  int* const p = Device::alloc<int>(0);
  assert(p == nullptr);
  Device::free(p);
}

TEST(DeviceAllocTest, AllocateZeroOnDevice) {
  kernel_AllocDevice_AllocateZeroOnDevice<<<1, 1>>>();
  check_cuda_errors();
}

// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


__global__ void kernel_assert_false() {
  assert(false);
}

TEST(CheckCudaErrorsTest, AssertInKernelDetectedOnHostInFileWithLongName) {
  ASSERT_NO_THROW(check_cuda_errors());

  kernel_assert_false<<<1, 1>>>();
  EXPECT_THROW({
    try {
# 42 "foo/0123456789/abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ/bazinga/bar.cu" 1
      check_cuda_errors();
    } catch(const CudaError& ex) {
      EXPECT_STREQ(
        ex.what(),
        // Error is truncated because of long file name
        "CUDA ERROR, detected at foo/0123456789/abcdefghijklmnopqrstuvwxyz"
        "/ABCDEFGHIJKLMNOPQRSTUVWXYZ/bazinga/bar.cu:42: 710 cudaErrorAs");
      throw;
    }
  }, CudaError);
}

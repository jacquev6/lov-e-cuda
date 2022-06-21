// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


__global__ void kernel_assert_false() {
  assert(false);
}

TEST(CheckCudaErrorsNoSyncTest, AssertInKernelDetectedOnHost) {
  ASSERT_NO_THROW(check_cuda_errors());

  kernel_assert_false<<<1, 1>>>();
  // At this point, `check_cuda_errors_no_sync` might or might not throw,
  // depending on how quickly the device (or stream) synchronizes (race condition).
  // So we need to synchronize it:
  cudaStreamSynchronize(cudaStreamDefault);
  // ... before calling `check_cuda_errors_no_sync`
  EXPECT_THROW(check_cuda_errors_no_sync(), CudaError);
}

// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


class ArrayView2DTest : public testing::Test {
 protected:
  ArrayView2DTest() : array(s1, s0, memory) {
    for (unsigned i = 0; i != s1 * s0; ++i) {
      memory[i] = 3 * i;
    }
  }

  static const unsigned s1 = 4;
  static const unsigned s0 = 3;
  int memory[s1 * s0];  // NOLINT(runtime/arrays)
  ArrayView2D<Host, int> array;
};

const unsigned ArrayView2DTest::s1;
const unsigned ArrayView2DTest::s0;

TEST_F(ArrayView2DTest, GetSizes) {
  EXPECT_EQ(array.s0(), s0);
  EXPECT_EQ(array.s1(), s1);
}

TEST_F(ArrayView2DTest, Data) {
  EXPECT_EQ(array.data_for_legacy_use(), memory);
}

TEST_F(ArrayView2DTest, Index) {
  // array[i1][i0] == memory[i1 * s0 + i0]
  EXPECT_EQ(array[0][0], 0);
  EXPECT_EQ(array[0][1], 3);
  EXPECT_EQ(array[0][2], 6);
  EXPECT_EQ(array[1][0], 9);
  EXPECT_EQ(array[1][1], 12);
  EXPECT_EQ(array[1][2], 15);
  EXPECT_EQ(array[2][0], 18);
  EXPECT_EQ(array[2][1], 21);
  EXPECT_EQ(array[2][2], 24);
  EXPECT_EQ(array[3][0], 27);
  EXPECT_EQ(array[3][1], 30);
  EXPECT_EQ(array[3][2], 33);
}

TEST_F(ArrayView2DTest, ConvertToConst) {
  // Can convert to const
  ArrayView2D<Host, const int> const_array(array);

  // Can read from const
  EXPECT_EQ(const_array[2][1], 21);

  // Can't write to a const
  #if EXPECT_COMPILE_ERROR == __LINE__
    const_array[2][1] = 65;
  #endif

  // Can't convert back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
    ArrayView2D<Host, int> non_const_array(const_array);
  #endif
}

TEST_F(ArrayView2DTest, IndexOnce) {
  ArrayView1D<Host, const int> array_1 = array[1];
  EXPECT_EQ(array_1[0], 9);
  EXPECT_EQ(array_1[1], 12);
  EXPECT_EQ(array_1[2], 15);
}

TEST_F(ArrayView2DTest, Assign) {
  ArrayView2D<Host, int> other_array(0, 0, nullptr);

  // Can be assigned (with "non-owning pointer" semantics)
  other_array = array;
  EXPECT_EQ(other_array.s1(), s1);
  EXPECT_EQ(other_array.s0(), s0);
  EXPECT_EQ(other_array.data_for_legacy_use(), memory);
  EXPECT_EQ(other_array[3][1], 30);

  // Can't be assigned if dimensions don't match
  #if EXPECT_COMPILE_ERROR == __LINE__
    other_array = ArrayView1D<Host, int>(0, nullptr);
  #endif
  #if EXPECT_COMPILE_ERROR == __LINE__
    other_array = ArrayView3D<Host, int>(0, 0, 0, nullptr);
  #endif
}

TEST_F(ArrayView2DTest, AssignToConst) {
  ArrayView2D<Host, const int> const_a(0, 0, nullptr);

  // Can be assigned
  const_a = array;
  EXPECT_EQ(const_a[3][1], 30);

  // Can't be re-assigned to non-const
  ArrayView2D<Host, int> non_const_a(0, 0, nullptr);
  #if EXPECT_COMPILE_ERROR == __LINE__
    non_const_a = const_a;
  #endif
}

TEST_F(ArrayView2DTest, Copy) {
  int other_memory[s1 * s0];  // NOLINT(runtime/arrays)
  ArrayView2D<Host, int> other_array(s1, s0, other_memory);
  other_array[0][0] = 42;
  other_array[3][2] = 42;

  copy(array, other_array);

  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

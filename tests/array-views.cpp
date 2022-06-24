// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


class ArrayViewTest : public testing::Test {
 protected:
  ArrayViewTest() : array(s1, s0, memory) {
    for (unsigned i1 = 0; i1 != s1; ++i1) {
      for (unsigned i0 = 0; i0 != s0; ++i0) {
        memory[i1 * s0 + i0] = 13 * i1 + 17 * i0;
      }
    }
  }

  static const unsigned s1 = 4;
  static const unsigned s0 = 3;
  int memory[s1 * s0];  // NOLINT(runtime/arrays)
  ArrayView2D<Host, int> array;
};

const unsigned ArrayViewTest::s1;
const unsigned ArrayViewTest::s0;

TEST_F(ArrayViewTest, GetSizes) {
  EXPECT_EQ(array.s0(), s0);
  EXPECT_EQ(array.s1(), s1);
}

TEST_F(ArrayViewTest, Index) {
  // array[i1][i0] == memory[i1 * s0 + i0]
  EXPECT_EQ(array[0][0], 0);
  EXPECT_EQ(array[0][1], 17);
  EXPECT_EQ(array[0][2], 34);
  EXPECT_EQ(array[1][0], 13);
  EXPECT_EQ(array[1][1], 30);
  EXPECT_EQ(array[1][2], 47);
  EXPECT_EQ(array[2][0], 26);
  EXPECT_EQ(array[2][1], 43);
  EXPECT_EQ(array[2][2], 60);
  EXPECT_EQ(array[3][0], 39);
  EXPECT_EQ(array[3][1], 56);
  EXPECT_EQ(array[3][2], 73);
}

TEST_F(ArrayViewTest, ConvertToConst) {
  // Can convert to const
  ArrayView2D<Host, const int> const_array(array);

  // Can read from const
  EXPECT_EQ(const_array[2][1], 43);

  // Can't write to a const
  #if EXPECT_COMPILE_ERROR == __LINE__
    const_array[2][1] = 65;
  #endif

  // Cant' convert back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
    ArrayView2D<Host, int> non_const_array(const_array);
  #endif
}

TEST_F(ArrayViewTest, IndexOnce) {
  ArrayView1D<Host, const int> array_1 = array[1];
  EXPECT_EQ(array_1[0], 13);
  EXPECT_EQ(array_1[1], 30);
  EXPECT_EQ(array_1[2], 47);
}

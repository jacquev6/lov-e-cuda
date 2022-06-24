// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


class ArrayViewTest : public testing::Test {
 protected:
  ArrayViewTest() : m(s1, s0, memory) {
    for (unsigned i1 = 0; i1 != s1; ++i1) {
      for (unsigned i0 = 0; i0 != s0; ++i0) {
        memory[i1 * s0 + i0] = 13 * i1 + 17 * i0;
      }
    }
  }

  static const unsigned s1 = 4;
  static const unsigned s0 = 3;
  int memory[s1 * s0];  // NOLINT(runtime/arrays)
  ArrayView2D<Host, int> m;
};

const unsigned ArrayViewTest::s1;
const unsigned ArrayViewTest::s0;

TEST_F(ArrayViewTest, GetSizes) {
  EXPECT_EQ(m.s0(), s0);
  EXPECT_EQ(m.s1(), s1);
}

TEST_F(ArrayViewTest, Index) {
  // m[i1][i0] == memory[i1 * s0 + i0]
  EXPECT_EQ(m[0][0], 0);
  EXPECT_EQ(m[0][1], 17);
  EXPECT_EQ(m[0][2], 34);
  EXPECT_EQ(m[1][0], 13);
  EXPECT_EQ(m[1][1], 30);
  EXPECT_EQ(m[1][2], 47);
  EXPECT_EQ(m[2][0], 26);
  EXPECT_EQ(m[2][1], 43);
  EXPECT_EQ(m[2][2], 60);
  EXPECT_EQ(m[3][0], 39);
  EXPECT_EQ(m[3][1], 56);
  EXPECT_EQ(m[3][2], 73);
}

TEST_F(ArrayViewTest, ConvertToConst) {
  ArrayView2D<Host, const int> cm(m);
  EXPECT_EQ(cm[2][1], 43);

  // Can't write to a const
  #if EXPECT_COMPILE_ERROR == __LINE__
    cm[2][1] = 65;
  #endif

  // Cant' convert back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
    ArrayView2D<Host, int> ncm(cm);
  #endif
}

TEST_F(ArrayViewTest, IndexOnce) {
  ArrayView1D<Host, const int> m1 = m[1];
  EXPECT_EQ(m1[0], 13);
  EXPECT_EQ(m1[1], 30);
  EXPECT_EQ(m1[2], 47);
}

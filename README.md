<!-- Copyright 2022 Vincent Jacques -->
<!-- Copyright 2022 Laurent Cabaret -->

*Lov-e-cuda* is a single-file, header-only, C++ library providing basic utilities for CUDA programming.
It aims at imposing very little disruption compared to "traditional" CUDA programming,
while providing the comfort of [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)
and utilities to facilitate [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).

# Contents

## Error checking

Do you sometime forget to check for CUDA errors?
Are you tired of writing the same error checking code again and again?

*Lov-e-cuda* provides a set of simple function-like macros to check the return value of a cuda API call or the last CUDA error.

## Memory management

Is your code filled with `cudaMalloc`s, `cudaFree`s, `cudaMallocHost`s?
Do they sometime not match, leading to [undefined behavior](https://en.wikipedia.org/wiki/Undefined_behavior), crashes and memory leaks?
Do they mix with host-side `malloc`s and `new`s in an unpleasantly inconsistent way?
Do you sometime forget the `* sizeof(float)` in the argument of a `malloc` or `cudaMalloc`?

<!--
@todo Investigate the different types of memory on the host
- "normal" memory
- pinned memory
- zero-copy pinned memory
- what else?
@todo Investigate the different types of memory on the device
- "normal" memory
- shared memory
- what else?
-->

*Lov-e-cuda* provides consistent utilities to allocate and free memory on the host and on the device,
with an homogeneous API drop-in replacement for all these (de-)allocations functions.
All layers are templated on the type of data allocated, to make your code more DRY.

Similarly, *Lov-e-cuda* provides homogeneous replacements for the `cudaMemcpy`/`cudaMemset` and `memcpy`/`memset` functions.

Most importantly, it also provides RAII classes to make sure each allocation is matched with a deallocation.

## Multi-dimensional-array-like memory access

At the conceptual level, most parallel computing projects manipulate multi-dimensional arrays.
But in the code, this abstraction is often implemented using explicit computation of single-dimensional indexes like `data[i * width + j]`.
*Lov-e-cuda* provides template classes to "view" an area of memory as a multi-dimensional array.
This view is accessed like `data[i][j]`.
Thanks to inlining and return-value optimization, the performance cost is negligible when compiled with full optimization.
This increased abstraction level comes with the added benefit of boundary checks on each individual index (which can be deactivated by defining the `NDEBUG` C++ preprocessor macro).

## Grid and blocks

Are you tired of writing `const int x = blockIdx.x * blockDim.x + threadIdx.x`? (and `y`, and `z`)
Do you sometime forget that your data size might not be a perfect multiple of the number of threads in your CUDA blocks?
Do you have to check if blocks or threads come first in the kernel call configuration? (Is it `kernel<<<blocks, threads>>>` or `kernel<<<threads, blocks>>>`?)

*Lov-e-cuda* provides utilities to avoid repeating computations like `dim3 blocks((width + BLOCKDIM_X - 1) / BLOCKDIM_X)` (correct even when `width % BLOCKDIM_X != 0`), to call kernels and to retrieve `blockIdx.x * blockDim.x + threadIdx.x` in the kernels in a readable and efficient way.

# Authors

*Lov-e-cuda* was sponsored by [Laurent Cabaret](https://cabaretl.pages.centralesupelec.fr/cv/) from the [MICS](http://www.mics.centralesupelec.fr/) and written by [Vincent Jacques](https://vincent-jacques.net).

# Licensing and citation

*Lov-e-cuda* is licensed under the quite permissive MIT license.

Whenever appropriate, we kindly ask that you cite *Lov-e-cuda* in the following way: @todo Add BibTeX entry.
This is particularly traditional in the academic tradition, as denoted *e.g.* in the doc for [matplotlib](https://matplotlib.org/3.1.1/citing.html) and [GNU parallel](https://git.savannah.gnu.org/cgit/parallel.git/tree/doc/citation-notice-faq.txt).

# Quick start

*Lov-e-cuda* is a single-file header-only library so getting started is very simple:
- download `lov-e.hpp`
- put it in your include path, for example in `/usr/local/include` or simply besides your code
- add `#include <lov-e.hpp>` in your files

Then see the "User manual" and "Examples" sections below.

# Examples

Here are a few examples we provide for you to to see how *Lov-e-cuda* can be applied.
For each of them, we provide one version that uses *Lov-e-cuda*, and one that doesn't.
We use the same, somewhat pedantic, coding standards in all examples so that they differ only by the introduction of *Lov-e-cuda*.

Comparing the versions without and with *Lov-e-cuda* shows how the abstractions it provides simplify the code.

Comparing their runtime performance proves that *Lov-e-cuda* is neutral on that aspect.
Note that this is true *only* when compiling with -DNDEBUG to deactivate `assert`s.
On the other hand, when `assert`s are activated, code using *Lov-e-cuda* can be slower.
That is because it checks that all indexes are within boundaries, and so it's *safer*.

## Descriptions

### Mandelbrot

Based on this [NVidia blog article by Andy Adinets about dynamic parallelism](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/), we provide two examples that produce a 16384x16384 image of the Mandelbrot set: one with dynamic parallelism and one without.

## Runtime performance

All performance below have been measured on code compiled with `-O3 -DNDEBUG`.

<!-- BEGIN GENERATED SECTION: examples-performance-table -->

| Example | Without *Lov-e-cuda* | With *Lov-e-cuda* |
| --- | --- | --- |
| Mandelbrot<br>(static parallelism) | 183 ms *i.e.* 1464 Mpix/s | 191 ms *i.e.* 1407 Mpix/s |
| Mandelbrot<br>(dynamic parallelism) | 43 ms *i.e.* 6272 Mpix/s | 36 ms *i.e.* 7515 Mpix/s |

<!-- END GENERATED SECTION: examples-performance-table -->

# User manual

## Multi-dimensional arrays

Instead of:

<!-- BEGIN GENERATED SECTION: user-manual-snippet(bad-multi-dim-arrays) -->
<!-- END GENERATED SECTION: user-manual-snippet(bad-multi-dim-arrays) -->

<!-- @todo Document that indexes and sizes are in order sN -> ... -> s1 -> s0
(To ensure that the right-most index is always i0. Useful when doing b = a[i1]; c = b[i0]) -->

## Grid and blocks

<!-- @todoc -->

## Lower-level Memory management

<!-- @todo Document how to allocate and memset non-trivial types -->

## Error checking

<!-- @todoc -->

# Development

## Dependencies

*Lov-e-cuda* is developed in a controlled environment using [Docker](https://www.docker.com/) and the Docker image built automatically by `make.sh` from `builder/Dockerfile`.
Contributors only need reasonably recent versions of Python, Bash, and Docker to run `./run-development-cycle.py` to run all automated tests.

## Tests

There are a few automated tests:

- unit tests using [Google Test](https://google.github.io/googletest/) are in the `tests` directory
- some of these tests use the custom tool `builder/make-non-compilation-tests-deps.py` to test for expected compile-time errors
- the code snippet in the "User manual" section above come from `tests/user-manual.cu` and are copied to this `README.md` file by teh `run-development-cycle.py` script

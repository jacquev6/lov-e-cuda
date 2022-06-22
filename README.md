<!-- Copyright 2022 Vincent Jacques -->
<!-- Copyright 2022 Laurent Cabaret -->

*Lov-e-cuda* is a single-file, header-only, C++ library providing basic utilities for CUDA programming.
It aims at imposing very little disruption compared to "traditional" CUDA programming,
while providing the comfort of [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)
and utilities to facilitate [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).

# Contents

<!-- @todo Add section about error checking -->
<!-- @todo What about using the type checker to ensure Device pointers are never dereferenced on the Host and reciprocally? -->

## Memory management

Is your code filled with `cudaMalloc`s, `cudaFree`s, `cudaMallocHost`s?
Do they sometime not match, leading to [undefined behavior](), crashes and memory leaks?
Do they mix with `malloc`s and `new`s in an inconsistent way?
Do you sometime forget the ` * sizeof(float)` in the argument of a `malloc` or `cudaMalloc`?

*Lov-e-cuda* provides consistent utilities to allocate and free memory on the host (paged and pinned) and on the device (@todo Investigate the different types of memory on the device).
The "low abstraction level" layer provides an homogeneous API drop-in replacement for all these (de-)allocations functions.
The "medium abstraction level" layer brings in RAII to ensure de-allocations match allocations exactly.
All layers are templated on the type of data allocated, to make your code more DRY.

Similarly, *Lov-e-cuda* provides homogeneous replacements for the `cudaMemcpy` and `memcpy` functions.

## Multi-dimensional-array-like memory access

At the conceptual level, most parallel computing projects manipulate multi-dimensional arrays.
But in the code, this abstraction is often implemented using explicit computation of single-dimensional indexes like `data[i * width + j]`.
*Lov-e-cuda* provides template classes to "view" an area of memory as a multi-dimensional array.
This view is accessed like `data[i][j]`.
Thanks to inlining and return-value optimization, the performance cost is negligible (when compiled with full optimization).
This increased abstraction level comes with the added benefit of boundary checks on each individual index (which can be deactivated by defining the `NDEBUG` C++ preprocessor macro).

## Grid and blocks

Are you tired of writing `const int x = blockIdx.x * blockDim.x + threadIdx.x`? (and `y`, and `z`)
Do you sometime forget that your data size might not be a perfect multiple of the number of threads in your CUDA blocks?
Do you have to check if blocks or threads come first in the kernel call configuration? (Is it `kernel<<<block, threads>>>` or `kernel<<<threads, blocks>>>`?)

*Lov-e-cuda* provides utilities to avoid repeating computations like `dim3 blocks((width + BLOCKDIM_X - 1) / BLOCKDIM_X)` (correct even when `width % BLOCKDIM_X != 0`), to call kernels and to retrieve `blockIdx.x * blockDim.x + threadIdx.x` in the kernels in a readable and efficient way.

# Authors

*Lov-e-cuda* was sponsored by Laurent Cabaret (@todo Add contact info) and written by [Vincent Jacques](https://vincent-jacques.net).

# Licensing and citation

*Lov-e-cuda* is licensed under the quite permissive MIT license.

Whenever appropriate, we kindly ask that you cite *Lov-e-cuda* in the following way: @todo Add BibTeX entry

# Quick start

*Lov-e-cuda* is a single-file header-only library so getting started is very simple:
- download `lov-e.hpp`
- put it in your include path, for example in `/usr/local/include` or simply besides your code
- add `#include <lov-e.hpp>` in your files

Then see the "User manual" and "Examples" sections below.

# Examples

For each example, we provide one version that uses *Lov-e-cuda*, and one that doesn't.
We use the same, somewhat pedantic, coding standards in all examples so that they differ only by the introduction of *Lov-e-cuda*.

Comparing the versions without and with *Lov-e-cuda* shows how the abstractions it provides simplify the code.

Comparing their runtime performance proves that *Lov-e-cuda* is neutral on that aspect.
*Note* that this is true only when compiling with -DNDEBUG.
On the other hand, when `assert`s are activated, code using *Lov-e-cuda* can be significantly slower.
That is because it checks that all indexes are within boundaries, and so it's *safer*.

## Descriptions

### Mandelbrot

Based on this [NVidia blog article by Andy Adinets about dynamic parallelism](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/), we provide two examples that produce a 16384x16384 image of the Mandelbrot set: one with dynamic parallelism and one without.

## Runtime performance

<!-- @todo Generate this section from actual executions -->
<!-- @todo Average a few executions (10?) to get more significant numbers -->

All performance below have been measured on code compiled with `-O3 -DNDEBUG`.

| Example | Without *Lov-e-cuda* | With *Lov-e-cuda* |
| --- | --- | --- |
| Mandelbrot<br>(static parallelism) | 183 ms *i.e.* 1464 Mpix/s | 184 ms *i.e.* 1462 Mpix/s |
| Mandelbrot<br>(dynamic parallelism) | 43 ms *i.e.* 6274 Mpix/s | 39 ms *i.e.* 6920 Mpix/s |

# User manual

## Memory management

## Multi-dimensional-array-like memory access

<!-- @todo Document that indexes and sizes are in order sN -> ... -> s1 -> s0
(To ensure that the right-most index is always i0. Useful when doing b = a[i1]; c = b[i0]) -->

## Grid and blocks

# Development

## Dependencies

*Lov-e-cuda* is developed in a controlled environment using [Docker](https://www.docker.com/) and the Docker image built automatically by `make.sh` from `builder/Dockerfile`.
Contributors only need reasonably recent versions of Docker and Bash to run `./make.sh -j$(nproc)` to run all automated tests.
The few tests that actually use a GPU require that the NVidia runtime is installed. Otherwise, an explicit warning is printed.

## Tests

There are a few automated tests:

- unit tests using [Google Test](https://google.github.io/googletest/) are in the `tests` directory
- some of these tests use the custom tool `builder/@todo` to test for expected compile-time errors
- the examples in this `README.md` file are also extracted and run as automated tests by `builder/@todo`

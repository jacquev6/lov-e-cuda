# Copyright 2022 Vincent Jacques
# Copyright 2022 Laurent Cabaret

# https://stackoverflow.com/a/52603343/905845
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables
# https://stackoverflow.com/a/14391872/905845
MAKEFLAGS += --warn-undefined-variables

# https://stackoverflow.com/a/589300/905845
SHELL := /bin/bash -o pipefail -o errexit

############################
# Default top-level target #
############################

.PHONY: default
default: lint link tests

#############
# Inventory #
#############

root_directory := $(shell pwd)

# Source files
cuda_unit_test_source_files := $(wildcard tests/*.cu)
cpp_unit_test_source_files := $(filter-out tests/main.cpp,$(wildcard tests/*.cpp))
cuda_example_source_files := $(wildcard examples/*.cu)
lintable_source_files := lov-e.hpp tests/main.cpp $(cuda_unit_test_source_files) $(cpp_unit_test_source_files) $(cuda_example_source_files)

# Intermediate files
non_compilation_includes := $(patsubst %,build/deps/%.non-compilation.deps,$(cpp_unit_test_source_files) $(cuda_unit_test_source_files))

# Output files
object_files := \
    build/debug/tests/main.o \
    $(patsubst %.cu,build/debug/%.o,$(cuda_unit_test_source_files)) \
    $(patsubst %.cpp,build/debug/%.o,$(cpp_unit_test_source_files)) \
    $(patsubst %.cu,build/debug/%.o,$(cuda_example_source_files)) \
    $(patsubst %.cu,build/release/%.o,$(cuda_example_source_files))
binary_files := $(patsubst %.o,%,$(object_files))
cpplint_sentinel_files := $(patsubst %,build/cpplint/%.cpplint.ok,$(lintable_source_files))
# 'cuda-memcheck' fails on processes that 'assert' on device, so we don't run it on 'error-checking-*' tests
unit_test_sentinel_files := \
    $(patsubst %.cu,build/debug/%.plain.ok,$(cuda_unit_test_source_files)) \
    $(patsubst %.cu,build/debug/%.valgrind-memcheck.ok,$(cuda_unit_test_source_files)) \
    $(patsubst %.cu,build/debug/%.cuda-memcheck.ok,$(filter-out tests/error-checking-%.cu,$(cuda_unit_test_source_files))) \
    $(patsubst %.cpp,build/debug/%.plain.ok,$(cpp_unit_test_source_files)) \
    $(patsubst %.cpp,build/debug/%.valgrind-memcheck.ok,$(cpp_unit_test_source_files))
example_sentinel_files := $(patsubst %.cu,build/release/%.ok,$(cuda_example_source_files))

###############################
# Secondary top-level targets #
###############################

# Lint
# ====

.PHONY: lint
lint: cpplint

.PHONY: cpplint
cpplint: $(cpplint_sentinel_files)

build/cpplint/%.cpplint.ok: %
	@echo "cpplint $<"
	@mkdir -p $(dir $@)
	@cpplint --linelength=120 $< 2>&1 | tee $(patsubst %.ok,%.log,$@) | (grep -v "Done processing $<" || true)
	@touch $@


# Compile and link
# ================

.PHONY: compile
compile: $(object_files)

.PHONY: link
link: $(binary_files)

# Test
# ====

.PHONY: tests
tests: unit-tests non-compilation-tests

.PHONY: unit-tests
unit-tests: $(unit_test_sentinel_files)

$(foreach file,$(non_compilation_includes),$(eval include $(file)))

build/deps/%.non-compilation.deps: %
	@mkdir -p $(dir $@)
	@builder/make-non-compilation-tests-deps.py $^ >$@

# Examples
# ========

.PHONY: examples
examples: $(example_sentinel_files)

#################
# Generic rules #
#################

# Ah the pain of enabling "all" 'g++' warnings... https://stackoverflow.com/a/11720263/905845
gcc_flags := -std=c++11 -fopenmp -W -Wall -Wextra -Werror -pedantic -I/usr/local/cuda-11.2/targets/x86_64-linux/include

# Targets:
# - 52: Vincent's GeForce GTX TITAN X and Laurent's GeForce GTX 980 Ti
# - 75: Laurent's GeForce RTX 2080 Ti
# @todo Put targets in a list and generate 'nvcc_targets'
nvcc_targets := -arch=sm_75 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_75,code=sm_75
nvcc_flags := -std=c++11 -Xcompiler "-fopenmp -W -Wall -Wextra -Werror" $(nvcc_targets)

link_flags := -lgtest -lpng

# Debug
# =====

debug_flags := -g -O0

# Compile
build/debug/%.o: %.cu lov-e.hpp
	@echo "nvcc -dc $<"
	@mkdir -p $(dir $@)
	@nvcc -dc $(nvcc_flags) $(debug_flags) $< -o $@

build/debug/%.o: %.cpp lov-e.hpp
	@echo "g++ -c $<"
	@mkdir -p $(dir $@)
	@g++ -c  $(gcc_flags) $(debug_flags) $< -o $@

# Link test executables
build/debug/tests/%: build/debug/tests/%.o build/debug/tests/main.o
	@echo "nvcc -o $@"
	@mkdir -p $(dir $@)
	@nvcc $(nvcc_flags) $(debug_flags) $(link_flags) $^ -o $@

# Link examples executables
build/debug/examples/%: build/debug/examples/%.o
	@echo "nvcc -o $@"
	@mkdir -p $(dir $@)
	@nvcc $(nvcc_flags) $(debug_flags) $(link_flags) $^ -o $@

# Run
build/debug/%.plain.ok: build/debug/%
	@echo "$<"
	@mkdir -p $(patsubst %.ok,%.wd,$@)
	@(cd $(patsubst %.ok,%.wd,$@); $(root_directory)/$<) 2>&1 | tee $(patsubst %.ok,%.log,$@)
	@touch $@

# Run with Valgrind's memcheck
# Notes that Valgrind and CUDA aren't very friendly to each other, so a pretty brutal suppression file is used
# to hide false positives. Sadly it may well hide true positives as well. But this is better than nothing.
build/debug/%.valgrind-memcheck.ok: build/debug/%
	@echo "valgrind $<"
	@mkdir -p $(patsubst %.ok,%.wd,$@)
	@(cd $(patsubst %.ok,%.wd,$@); valgrind --tool=memcheck --verbose --leak-check=full --show-leak-kinds=definite,indirect,possible --errors-for-leak-kinds=definite,indirect,possible --error-exitcode=1 --gen-suppressions=all --suppressions=$(root_directory)/builder/valgrind-cuda.supp $(root_directory)/$<) 2>&1 | tee $(patsubst %.ok,%.log,$@)
	@touch $@

# Run with NVidia's 'cuda-memcheck'
build/debug/%.cuda-memcheck.ok: build/debug/%
	@echo "cuda-memcheck $<"
	@mkdir -p $(patsubst %.ok,%.wd,$@)
	@(cd $(patsubst %.ok,%.wd,$@); cuda-memcheck --tool memcheck --leak-check full --error-exitcode 1 $(root_directory)/$<) 2>&1 | tee $(patsubst %.ok,%.log,$@)
	@touch $@


# Release
# =======

gcc_release_flags := -march=native -mtune=native -O3 -DNDEBUG
nvcc_release_flags := -Xcompiler "-march=native -mtune=native" -O3 -DNDEBUG

# Compile
build/release/%.o: %.cu lov-e.hpp
	@echo "nvcc -dc $<"
	@mkdir -p $(dir $@)
	@nvcc -dc $(nvcc_flags) $(nvcc_release_flags) $< -o $@

build/release/%.o: %.cpp lov-e.hpp
	@echo "g++ -c $<"
	@mkdir -p $(dir $@)
	@g++ -c $(gcc_flags) $(gcc_release_flags) $< -o $@

# Link
build/release/%: build/release/%.o
	@echo "nvcc -o $@"
	@mkdir -p $(dir $@)
	@nvcc $(nvcc_flags) $(nvcc_release_flags) $(link_flags) $^ -o $@

# Run
build/release/%.ok: build/release/%
	@echo "$<"
	@mkdir -p $(patsubst %.ok,%.wd,$@)
	@(cd $(patsubst %.ok,%.wd,$@); $(root_directory)/$<) 2>&1 | tee $(patsubst %.ok,%.log,$@)
	@touch $@

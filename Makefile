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
default: lint tests examples

#############
# Inventory #
#############

root_directory := $(shell pwd)

# Source files
cuda_unit_test_source_files := $(wildcard tests/*.cu)
cpp_unit_test_source_files := $(wildcard tests/*.cpp)
cuda_example_source_files := $(wildcard examples/*.cu)
lintable_source_files := lov-e.hpp $(cuda_unit_test_source_files) $(cpp_unit_test_source_files)

# Intermediate files
cpplint_sentinel_files := $(patsubst %,build/cpplint/%.ok,$(lintable_source_files))
unit_test_sentinel_files := $(patsubst %.cu,build/debug/%.ok,$(cuda_unit_test_source_files)) $(patsubst %.cpp,build/debug/%.ok,$(cpp_unit_test_source_files))
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

build/cpplint/%.ok: %
	@echo "cpplint $<"
	@mkdir -p $(dir $@)
	@cpplint --linelength=120 $< | tee build/cpplint/$*.log
	@touch $@

# Test
# ====

.PHONY: tests
tests: unit-tests

.PHONY: unit-tests
unit-tests: $(unit_test_sentinel_files)

# Example
# =======

.PHONY: examples
examples: $(example_sentinel_files)

#################
# Generic rules #
#################

# Debug
# =====

# Compile
build/debug/%.o: %.cu lov-e.hpp
	@echo "nvcc -dc $<"
	@mkdir -p $(dir $@)
	@nvcc -dc -g -Xcompiler -fopenmp $< -o $@

build/debug/%.o: %.cpp lov-e.hpp
	@echo "g++ -c $<"
	@mkdir -p $(dir $@)
	@g++ -c -fopenmp -g -I/usr/local/cuda-11.2/targets/x86_64-linux/include $< -o $@

# Link
build/debug/%: build/debug/%.o
	@echo "nvcc -o $@"
	@mkdir -p $(dir $@)
	@nvcc -g -Xcompiler -fopenmp -lgtest_main -lgtest -lpng $^ -o $@

# Release
# =======

# Compile
build/release/%.o: %.cu lov-e.hpp
	@echo "nvcc -dc $<"
	@mkdir -p $(dir $@)
	@nvcc -dc -O3 -DNDEBUG -Xcompiler -fopenmp $< -o $@

build/release/%.o: %.cpp lov-e.hpp
	@echo "g++ -c $<"
	@mkdir -p $(dir $@)
	@g++ -c -fopenmp -O3 -DNDEBUG -I/usr/local/cuda-11.2/targets/x86_64-linux/include $< -o $@

# Link
build/release/%: build/release/%.o
	@echo "nvcc -o $@"
	@mkdir -p $(dir $@)
	@nvcc -O3 -DNDEBUG -Xcompiler -fopenmp -lgtest_main -lgtest -lpng $^ -o $@

# Run
# ===

build/%.ok: build/%
	@echo "$<"
	@mkdir -p build/$*-wd
	@(cd build/$*-wd; $(root_directory)/$<) | tee build/$*.log
	@touch $@

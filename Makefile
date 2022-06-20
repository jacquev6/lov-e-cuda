# Copyright 2022 Vincent Jacques
# Copyright 2022 Laurent Cabaret

# https://stackoverflow.com/a/52603343/905845
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

# https://stackoverflow.com/a/589300/905845
SHELL := /bin/bash -o pipefail -o errexit

############################
# Default top-level target #
############################

.PHONY: default
default: test lint

#############
# Inventory #
#############

# Source files
cuda_test_source_files := $(wildcard tests/*.cu)
cpp_test_source_files := $(wildcard tests/*.cpp)
all_source_files := lov-e.hpp $(cuda_test_source_files) $(cpp_test_source_files)

# Intermediate files
unit_test_sentinel_files := $(patsubst tests/%.cu,build/tests/%.ok,$(cuda_test_source_files)) $(patsubst tests/%.cpp,build/tests/%.ok,$(cpp_test_source_files))
cpplint_sentinel_files := $(patsubst %,build/cpplint/%.ok,$(all_source_files))

###############################
# Secondary top-level targets #
###############################

# Test
# ====

.PHONY: test
test: unit-tests

.PHONY: unit-tests
unit-tests: $(unit_test_sentinel_files)

build/tests/%.ok: build/bin/%
	@echo $<
	@mkdir -p $(dir $@)
	@$< | tee build/tests/$*.log
	@touch $@

build/bin/%: build/obj/%.o
	@echo "nvcc -o $@"
	@mkdir -p $(dir $@)
	@nvcc -g $^ -lgtest_main -lgtest -o $@

build/obj/%.o: tests/%.cu lov-e.hpp
	@echo "nvcc -dc $<"
	@mkdir -p $(dir $@)
	@nvcc -g -dc $< -o $@

build/obj/%.o: tests/%.cpp lov-e.hpp
	@echo "g++ -c $<"
	@mkdir -p $(dir $@)
	@g++ -g -c $< -I/usr/local/cuda-11.2/targets/x86_64-linux/include -o $@

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

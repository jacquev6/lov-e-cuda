#!/bin/bash
# Copyright 2022 Vincent Jacques
# Copyright 2022 Laurent Cabaret

set -o errexit


if ! diff -r builder build/builder >/dev/null 2>&1 || ! diff Makefile build/Makefile >/dev/null 2>&1
then
  rm -rf build
  mkdir build
  docker build --tag love-e-cuda-builder builder
  cp -r builder build
  cp Makefile build
fi


if test -f build/nvidia-docker-runtime.ok || docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1
then
  docker_gpu_options="--gpus all --env LOV_E_HAS_GPU=true"
  touch build/nvidia-docker-runtime.ok
else
  echo "************************************************************************"
  echo "** The NVidia Docker runtime does not seem to be properly configured. **"
  echo "** Tests that require a GPU will *not* be run.                        **"
  echo "************************************************************************"
  docker_gpu_options="--env LOV_E_HAS_GPU=false"
fi


docker run --rm \
  --volume $PWD:/project --workdir /project \
  --user $(id -u):$(id -g) `# Avoid creating files as 'root'` \
  $docker_gpu_options \
  --network none `# Ensure the repository is self-contained (except for the "docker build" phase)` \
  --volume "$PWD:/wd" --workdir /wd \
  love-e-cuda-builder \
    make "$@"

#!/bin/bash -xe

/opt/conda/envs/rose_conda_pytorch/bin/python -u main.py `< ${DL_PLATFORM_JOB_CONFIG_PATH}`

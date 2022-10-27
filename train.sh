#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

python train.py --config-path ./configs --config-name train_config.yaml

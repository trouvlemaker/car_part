#!/bin/bash
python inference.py \
--config-path ./configs/infer_config.json \
--image_path ./sample_image/ \
--output_dir ./sample_outputs/ \
--weights_path ./trained_model/kaflix-part-best.h5

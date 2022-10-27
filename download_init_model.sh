#!/bin/bash

model_name="aos_parts_init_model"
model_version="1"

soda-cli download-model -m ${model_name} -v ${model_version} -d /artifacts/${model_name}

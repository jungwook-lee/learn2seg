#!/bin/bash

source activate tensorflow_p36
rm -rf /home/ubuntu/output/unit_test
python scripts/train.py ~/git/learn2seg/configs/unet_unit_aws_v3.yaml

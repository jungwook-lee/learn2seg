#!/bin/bash

source activate tensorflow_p36
nohup python scripts/train.py ~/git/learn2seg/configs/unet_aws_v3.yaml &

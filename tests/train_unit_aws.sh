#!/bin/bash

rm -rf /home/jung/output/unit_test
python scripts/train.py ~/git/learn2seg/configs/unet_unit_aws.yaml

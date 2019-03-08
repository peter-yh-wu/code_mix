#!/bin/bash


python3 -u baseline.py --save-directory output/baseline/lr_1e-3
python3 -u baseline.py --save-directory output/baseline/lr_3e-4 --lr 3e-4
python3 -u baseline.py --save-directory output/baseline/lr_1e-4 --lr 1e-4
python3 -u baseline.py --save-directory output/baseline/lr_3e-5 --lr 3e-5
python3 -u baseline.py --save-directory output/baseline/lr_1e-5 --lr 1e-5
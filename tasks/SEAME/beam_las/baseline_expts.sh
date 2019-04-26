#!/bin/bash

python3 -u baseline.py --save-directory output/baseline/5k1k1k --max-train 5000 --max-dev 1000 --max-test 1000
python3 -u baseline.py --save-directory output/baseline/10-1-1k --max-train 10000 --max-dev 1000 --max-test 1000
python3 -u baseline.py --save-directory output/baseline/1k1k1k --max-train 1000 --max-dev 1000 --max-test 1000
python3 -u baseline.py --save-directory output/baseline/20k5k5k --max-train 20000 --max-dev 5000 --max-test 5000
python3 -u baseline.py --save-directory output/baseline/lr_1e-3
python3 -u baseline.py --save-directory output/baseline/lr_3e-4 --lr 3e-4
python3 -u baseline.py --save-directory output/baseline/lr_1e-4 --lr 1e-4
python3 -u baseline.py --save-directory output/baseline/lr_3e-5 --lr 3e-5
python3 -u baseline.py --save-directory output/baseline/lr_1e-5 --lr 1e-5
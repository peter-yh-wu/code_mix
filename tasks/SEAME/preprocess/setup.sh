#!/bin/bash

./get_mfcc.sh
# python3 -u preprocess_text.py
python3 -u split_data.py
python3 -u remove_empty.py
python3 -u remove_nonaudio.py
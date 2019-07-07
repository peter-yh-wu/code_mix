# Setup Instructions

- Put the .mfcc files in ```data/mfcc```
- ```cd cs_las```
- Run ```python3 main.py``` to train baseline LAS model

# Analyzing Results

- ```cd analyze_results```
- Put generated transcripts, e.g. ```submission.csv``` file, in e.g. ```results``` directory
- To calculate WER: ```python3 get_error_rates.py --mode wer --file submission.csv --save-directory results```
- To calculate CER: ```python3 get_error_rates.py --mode cer --file submission.csv --save-directory results```
- To calculate top-k CER: ```python3 get_error_rates.py --mode topk --file beam.csv --save-directory results```

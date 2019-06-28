# Setup Instructions

- Put the .mfcc files in ```data/mfcc```
- ```cd las```
- Run ```python3 main.py``` to train LAS model

# Analyzing Results

- ```cd analyze_results```
- Put generated transcripts, e.g. ```submission.csv``` file, in e.g. ```results``` directory
- To calculate WER: ```python3 get_error_rates --mode wer --file submission.csv --save-directory results```
- To calculate CER: ```python3 get_error_rates --mode cer --file submission.csv --save-directory results```
- To calculate top-k CER: ```python3 get_error_rates --mode topk --file beam.csv --save-directory results```

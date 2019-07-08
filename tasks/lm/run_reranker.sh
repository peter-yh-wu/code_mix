#!/usr/bin/env bash

python3 rerank.py --gpu_id 0 --lm-path tagalog_models/best_tagalog.pt --submission-csv data/submission_beam_5_v3.csv --dataset tagalog > rerank_tagalog_5_v3.txt
python3 rerank.py --gpu_id 0 --lm-path tagalog_models/best_tagalog.pt --submission-csv data/submission_beam_10_v3.csv --dataset tagalog > rerank_tagalog_10_v3.txt

python3 rerank.py --gpu_id 0 --lm-path tagalog_models/best_tagalog.pt --submission-csv data/submission_beam_5_v5.csv --dataset tagalog > rerank_tagalog_5_v5.txt
python3 rerank.py --gpu_id 0 --lm-path tagalog_models/best_tagalog.pt --submission-csv data/submission_beam_10_v5.csv --dataset tagalog > rerank_tagalog_10_v5.txt

python3 rerank.py --gpu_id 0 --lm-path tagalog_models/best_tagalog.pt --submission-csv data/submission_beam_5_v6.csv --dataset tagalog > rerank_tagalog_5_v6.txt
python3 rerank.py --gpu_id 0 --lm-path tagalog_models/best_tagalog.pt --submission-csv data/submission_beam_10_v6.csv --dataset tagalog > rerank_tagalog_10_v6.txt
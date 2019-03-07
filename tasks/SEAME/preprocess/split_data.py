'''
Generates train-dev-test split

To-do:
 - Add conversation data to split
'''

import os
import random

SEED = 0
TRAIN_FRAC = 0.8
DEV_FRAC = 0.1

TRAIN_IDS_FILE = 'train_ids.txt'
DEV_IDS_FILE = 'dev_ids.txt'
TEST_IDS_FILE = 'test_ids.txt'

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERVIEW_WAV_DIR = os.path.join(parent_dir, 'data/interview/wav')

SPLIT_DIR = os.path.join(parent_dir, 'split')
if not os.path.exists(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)
train_ids_path = os.path.join(SPLIT_DIR, TRAIN_IDS_FILE)
dev_ids_path = os.path.join(SPLIT_DIR, DEV_IDS_FILE)
test_ids_path = os.path.join(SPLIT_DIR, TEST_IDS_FILE)

wav_files = os.listdir(INTERVIEW_WAV_DIR)
wav_files = [f for f in wav_files if f.endswith('.wav')]

ids = [f[:-4] for f in wav_files]

random.seed(SEED)
random.shuffle(ids)

num_samples = len(ids)
num_train = TRAIN_FRAC*num_samples
num_dev = DEV_FRAC*num_samples

train_ids = ids[:num_train]
dev_ids = ids[num_train:num_train+num_dev]
test_ids = ids[num_train+num_dev:]

with open(train_ids_path, 'w+') as ouf:
    for fid in train_ids:
        os.write('%s\n' % fid)
with open(dev_ids_path, 'w+') as ouf:
    for fid in dev_ids:
        os.write('%s\n' % fid)
with open(test_ids_path, 'w+') as ouf:
    for fid in test_ids:
        os.write('%s\n' % fid)
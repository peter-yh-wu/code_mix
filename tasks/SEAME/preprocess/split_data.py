'''
Generates train-dev-test split

To-do:
 - Add conversation data to split

Peter Wu
peterw1@andrew.cmu.edu
'''

import os
import pickle
import random


SEED = 0
TRAIN_FRAC = 0.8
DEV_FRAC = 0.1

TRAIN_IDS_FILE = 'train_ids.txt'
DEV_IDS_FILE = 'dev_ids.txt'
TEST_IDS_FILE = 'test_ids.txt'
TRAIN_YS_FILE = 'train_ys.txt'
DEV_YS_FILE = 'dev_ys.txt'
TEST_YS_FILE = 'test_ys.txt'

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERVIEW_WAV_DIR = os.path.join(parent_dir, 'data/interview/wav')

SPLIT_DIR = os.path.join(parent_dir, 'split')
if not os.path.exists(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)
train_ids_path = os.path.join(SPLIT_DIR, TRAIN_IDS_FILE)
dev_ids_path = os.path.join(SPLIT_DIR, DEV_IDS_FILE)
test_ids_path = os.path.join(SPLIT_DIR, TEST_IDS_FILE)
train_ys_path = os.path.join(SPLIT_DIR, TRAIN_YS_FILE)
dev_ys_path = os.path.join(SPLIT_DIR, DEV_YS_FILE)
test_ys_path = os.path.join(SPLIT_DIR, TEST_YS_FILE)

wav_files = os.listdir(INTERVIEW_WAV_DIR)
wav_files = [f for f in wav_files if f.endswith('.wav')]

ids = [f[:-4] for f in wav_files]

random.seed(SEED)
random.shuffle(ids)

num_samples = len(ids)
num_train = int(TRAIN_FRAC*num_samples)
num_dev = int(DEV_FRAC*num_samples)

train_ids = ids[:num_train]
dev_ids = ids[num_train:num_train+num_dev]
test_ids = ids[num_train+num_dev:]

with open(train_ids_path, 'w+') as ouf:
    for fid in train_ids:
        ouf.write('%s\n' % fid)
with open(dev_ids_path, 'w+') as ouf:
    for fid in dev_ids:
        ouf.write('%s\n' % fid)
with open(test_ids_path, 'w+') as ouf:
    for fid in test_ids:
        ouf.write('%s\n' % fid)

INTERVIEW_TEXT_DIR = os.path.join(parent_dir, 'data/interview/transcript/phaseI')
txt_files = os.listdir(INTERVIEW_TEXT_DIR)
txt_files = [f for f in txt_files if f.endswith('.txt')]
txt_paths = [os.path.join(INTERVIEW_TEXT_DIR, f) for f in txt_files]
all_ys = {}
for f in txt_paths:
    with open(f, 'r') as inf:
        lines = inf.readlines()
    lines = [l.strip() for l in lines]
    for l in lines:
        tokens = l.split()
        fid = tokens[0]+'_'+tokens[1]+'_'+tokens[2]
        start_i = len(fid)+1
        y_label = l[start_i:]
        all_ys[fid] = y_label
train_ys = [all_ys[fid] for fid in train_ids]
dev_ys = [all_ys[fid] for fid in dev_ids]
test_ys = [all_ys[fid] for fid in test_ids]

with open(train_ys_path, 'w+') as ouf:
    for y in train_ys:
        ouf.write('%s\n' % y)
with open(dev_ys_path, 'w+') as ouf:
    for y in dev_ys:
        ouf.write('%s\n' % y)
with open(test_ys_path, 'w+') as ouf:
    for y in test_ys:
        ouf.write('%s\n' % y)

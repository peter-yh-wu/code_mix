'''
Removes non-spoken text

Peter Wu
peterw1@andrew.cmu.edu
'''

import os

TRAIN_YS_FILE = 'train_ys.txt'
DEV_YS_FILE = 'dev_ys.txt'
TEST_YS_FILE = 'test_ys.txt'

def remove_lang_tags(ys):
    new_ys = []
    for y in ys:
        new_y_list = y.split()
        if new_y_list[0] == 'CS' or new_y_list[0] == 'EN' or new_y_list[0] == 'ZH':
            new_y_list = new_y_list[1:]
        new_y = ' '.join(new_y_list)
        new_ys.append(new_y)
    return new_ys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR = os.path.join(parent_dir, 'split')
if not os.path.exists(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)
train_ys_path = os.path.join(SPLIT_DIR, TRAIN_YS_FILE)
dev_ys_path = os.path.join(SPLIT_DIR, DEV_YS_FILE)
test_ys_path = os.path.join(SPLIT_DIR, TEST_YS_FILE)

with open(train_ys_path, 'r') as inf:
    train_ys = inf.readlines()
train_ys = [y.strip() for y in train_ys]
with open(dev_ys_path, 'r') as inf:
    dev_ys = inf.readlines()
dev_ys = [y.strip() for y in dev_ys]
with open(test_ys_path, 'r') as inf:
    test_ys = inf.readlines()
test_ys = [y.strip() for y in test_ys]

new_train_ys = remove_lang_tags(train_ys)
new_dev_ys = remove_lang_tags(dev_ys)
new_test_ys = remove_lang_tags(test_ys)

with open(train_ys_path, 'w+') as ouf:
    for y in new_train_ys:
        ouf.write('%s\n' % y)
with open(dev_ys_path, 'w+') as ouf:
    for y in new_dev_ys:
        ouf.write('%s\n' % y)
with open(test_ys_path, 'w+') as ouf:
    for y in new_test_ys:
        ouf.write('%s\n' % y)
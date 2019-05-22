'''
mk switch lid
'''

import os

def is_chinese_char(ch):
    curr_ord = ord(ch)
    if 11904 <= curr_ord and curr_ord <= 12031:
        return True
    elif 12352 <= curr_ord and curr_ord <= 12543:
        return True
    elif 13056 <= curr_ord and curr_ord <= 19903:
        return True
    elif 19968 <= curr_ord and curr_ord <= 40959:
        return True
    elif 63744 <= curr_ord and curr_ord <= 64255:
        return True
    elif 65072 <= curr_ord and curr_ord <= 65103:
        return True
    elif 194560 <= curr_ord and curr_ord <= 195103:
        return True
    else:
        return False

def get_lid(ch):
    '''Return 0 for english char, 1 for chinese char'''
    if is_chinese_char(ch):
        return 1
    return 0

def get_lids(s):
    '''returns a string'''
    lid_str = ''
    if len(s) == 0:
        return lids
    for ch in s:
        lid_str += str(get_lid(ch))
    return lid_str

def get_switch_lids(s):
    '''returns a list of ints'''
    lids = []
    if len(s) == 0:
        return lids
    prev_lid = get_lid(s[0])
    lids.append(prev_lid)
    for ch in s:
        if ch != ' ':
            curr_lid = get_lid(ch)
            if curr_lid != prev_lid:
                lids.append(curr_lid)
                prev_lid = curr_lid
    return lids

def mk_lid_txt(in_path, out_path):
    with open(in_path, 'r', encoding="utf-8")  as inf:
        lines = inf.readlines()
    lids_strs = []
    for l in lines:
        l = l.strip()
        lid_str = get_lids(l)
        lids_strs.append(lid_str)
    with open(out_path, 'w+') as ouf:
        for l in lids_strs:
            ouf.write('%s\n' % l)

def mk_switch_lid_txt(in_path, out_path):
    with open(in_path, 'r')  as inf:
        lines = inf.readlines()
    all_lids = []
    for l in lines:
        l = l.strip()
        lids = get_switch_lids(l)
        all_lids.append(lids)
    lids_strs = []
    for lids in all_lids:
        lids_str = ' '.join([str(lid) for lid in lids])
        lids_strs.append(lids_str)
    with open(out_path, 'w+') as ouf:
        for l in lids_strs:
            ouf.write('%s\n' % l)

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_dir  = os.path.join(parent_dir, 'split')
    train_txt_path = os.path.join(split_dir, 'train_ys.txt')
    dev_txt_path = os.path.join(split_dir, 'dev_ys.txt')
    test_txt_path = os.path.join(split_dir, 'test_ys.txt')

    train_lid_path = os.path.join(split_dir, 'train_lids.txt')
    dev_lid_path = os.path.join(split_dir, 'dev_lids.txt')
    test_lid_path = os.path.join(split_dir, 'test_lids.txt')

    mk_lid_txt(train_txt_path, train_lid_path)
    mk_lid_txt(dev_txt_path, dev_lid_path)
    mk_lid_txt(test_txt_path, test_lid_path)

    '''
    train_switch_lid_path = os.path.join(split_dir, 'train_switch_lids.txt')
    dev_switch_lid_path = os.path.join(split_dir, 'dev_switch_lids.txt')
    test_switch_lid_path = os.path.join(split_dir, 'test_switch_lids.txt')

    mk_switch_lid_txt(train_txt_path, train_switch_lid_path)
    mk_switch_lid_txt(dev_txt_path, dev_switch_lid_path)
    mk_switch_lid_txt(test_txt_path, test_switch_lid_path)
    '''

if __name__ == '__main__':
    main()

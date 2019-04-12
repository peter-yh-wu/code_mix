'''
Script to convert hanzi characters in text data to pinyin

excerpt from text data files:
37NC45MBP_0101	3410661	3414971	and  then  she  left  her  childhood  church  因为  她  不  喜欢  那个
37NC45MBP_0101	3444961	3446481	oh  K.  明白
37NC45MBP_0101	3492671	3494561	cos  she  went  on  to  大学  la
37NC45MBP_0101	3565211	3568511	okay  今天  下午  想  吃  什么  what  do  you  feel  like  eating
37NC45MBP_0101	3576711	3581181	oh  你  在  二  零  吆  零  年  有什么  er  what  what  resolutions  do  you  have

Peter Wu
peterw1@andrew.cmu.edu
'''

import os
# from pypinyin import pinyin, lazy_pinyin, Style

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERVIEW_TEXT_DIR = os.path.join(parent_dir, 'data/interview/transcript/phaseI')
CONVO_TEXT_DIR = os.path.join(parent_dir, 'data/conversation/transcript/phaseI')
# INTERVIEW_PINYIN_DIR = os.path.join(parent_dir, 'data/interview/transcript_pinyin/phaseI')
# CONVO_PINYIN_DIR = os.path.join(parent_dir, 'data/conversation/transcript_pinyin/phaseI')
NEW_INTERVIEW_TEXT_DIR = os.path.join(parent_dir, 'data/interview/transcript_clean/phaseI')
NEW_CONVO_TEXT_DIR = os.path.join(parent_dir, 'data/conversation/transcript_clean/phaseI')

def hanzi_to_pinyin(s):
    '''Returns string with all chinese characters turned to pinyin
    
    Args:
        s: word, assumes characters are not space separated

    Returns:
        space-separated string of pinyin
    '''
    return ' '.join(lazy_pinyin(s))

def mk_pinyin_file(in_path, out_path):
    '''writes pinyin version of in_path txt file to out_path'''
    with open(in_path, 'r') as inf:
        lines = inf.readlines()
    new_lines = []
    for l in lines:
        words = l.strip().split()
        new_words = []
        for w in words:
            new_words.append(hanzi_to_pinyin(w))
        new_lines.append(' '.join(new_words))
    with open(out_path, 'w+') as ouf:
        for l in new_lines:
            ouf.write('%s\n' % l)

def mk_pinyin_files(in_dir, out_dir):
    '''writes pinyin versions of all txt files in in_dir to out_dir'''
    files = os.listdir(in_dir)
    files = [f for f in files if f.endswith('.txt')]
    out_files = [f[:-4]+'_pinyin.txt' for f in files]
    in_paths = [os.path.join(in_dir, f) for f in files]
    out_paths = [os.path.join(out_dir, f) for f in out_files]
    for inp, oup in zip(in_paths, out_paths):
        mk_pinyin_file(inp, oup)

'''
if not os.path.exists(INTERVIEW_PINYIN_DIR):
    os.makedirs(INTERVIEW_PINYIN_DIR)
if not os.path.exists(CONVO_PINYIN_DIR):
    os.makedirs(CONVO_PINYIN_DIR)
mk_pinyin_files(INTERVIEW_TEXT_DIR, INTERVIEW_PINYIN_DIR)
mk_pinyin_files(CONVO_TEXT_DIR, CONVO_PINYIN_DIR)
'''

rm_list = ['[', ']', '［', '］', '`', '"', '#', '%', '\x7f', '\x84', '\x89', '\x90', '\x9d', '\x9f', '¡', '§', '·', '™', '\u3000']

def clean_w(w):
    new_w = w
    for c in rm_list:
        new_w = new_w.replace(c, '')
    new_w = new_w.replace('。', '.')
    new_w = new_w.replace('，', ',')
    new_w = new_w.replace('．', '.')
    new_w = new_w.replace('？', '?')
    new_w = new_w.replace('ａ', 'a')
    new_w = new_w.replace('ｂ', 'b')
    new_w = new_w.replace('ｃ', 'c')
    new_w = new_w.replace('ｄ', 'd')
    new_w = new_w.replace('ｅ', 'e')
    new_w = new_w.replace('ｉ', 'i')
    new_w = new_w.replace('ｍ', 'm')
    new_w = new_w.replace('ｎ', 'n')
    new_w = new_w.replace('ｏ', 'o')
    new_w = new_w.replace('ｐ', 'p')
    new_w = new_w.replace('ｒ', 'r')
    new_w = new_w.replace('ｓ', 's')
    new_w = new_w.replace('ｔ', 't')
    new_w = new_w.replace('ｕ', 'u')
    new_w = new_w.replace('ｖ', 'v')
    new_w = new_w.replace('=', '-')
    new_w = new_w.replace('(ppo)', '')
    new_w = new_w.replace('(ppc)', '')
    new_w = new_w.replace('(ppb)', '')
    new_w = new_w.replace('(ppl)', '')
    new_w = new_w.replace('[ppo]', '')
    new_w = new_w.replace('[ppc]', '')
    new_w = new_w.replace('[ppb]', '')
    new_w = new_w.replace('[ppl]', '')
    return new_w
    
def mk_clean_file(in_path, out_path):
    with open(in_path, 'r') as inf:
        lines = inf.readlines()
    new_lines = []
    for l in lines:
        words = l.strip().split()
        new_words = []
        for w in words:
            new_words.append(clean_w(w))
        new_lines.append(' '.join(new_words))
    with open(out_path, 'w+') as ouf:
        for l in new_lines:
            ouf.write('%s\n' % l)

def mk_clean_files(in_dir, out_dir):
    files = os.listdir(in_dir)
    files = [f for f in files if f.endswith('.txt')]
    out_files = [f[:-4]+'.txt' for f in files]
    in_paths = [os.path.join(in_dir, f) for f in files]
    out_paths = [os.path.join(out_dir, f) for f in out_files]
    for inp, oup in zip(in_paths, out_paths):
        mk_clean_file(inp, oup)

if not os.path.exists(NEW_INTERVIEW_TEXT_DIR):
    os.makedirs(NEW_INTERVIEW_TEXT_DIR)
if not os.path.exists(NEW_CONVO_TEXT_DIR):
    os.makedirs(NEW_CONVO_TEXT_DIR)
mk_clean_files(INTERVIEW_TEXT_DIR, NEW_INTERVIEW_TEXT_DIR)
mk_clean_files(CONVO_TEXT_DIR, NEW_CONVO_TEXT_DIR)
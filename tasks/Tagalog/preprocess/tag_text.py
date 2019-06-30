'''
Creates language id files for all text files in text directory
'''

import enchant
# import langid
import os

# from langdetect import detect

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    txt_dir = os.path.join(data_dir, 'text')
    txt_files = os.listdir(txt_dir)
    txt_files = [f for f in txt_files if f.endswith('.txt')]
    txt_paths = [os.path.join(txt_dir, f) for f in txt_files]

    lids_dir = os.path.join(data_dir, 'lids')
    if not os.path.exists(lids_dir):
        os.makedirs(lids_dir)

    d = enchant.Dict("en_US")

    for file_i, txt_path in enumerate(txt_paths):
        txt_file = txt_files[file_i]
        fid = txt_file[:-4]
        txt_file = fid+'_lids.txt'
        lids_path = os.path.join(lids_dir, txt_file)

        with open(txt_path, 'r') as inf:
            lines = inf.readlines()

        new_lines = []
        for line_i, l in enumerate(lines):
            l = l.strip()
            l_list = l.split()

            lids = []
            for w in l_list[1:]:
                if d.check(w):
                    lid = 'en'
                else:
                    # lid1 = langid.classify(w)[0]
                    # lid2 = detect(w)
                    # lid = lid2
                    lid = 'tl'
                lids.append(lid)

            new_l_list = [l_list[0]]+lids
            new_l = ' '.join(new_l_list)
            new_lines.append(new_l)

            if (line_i+1) % 1000 == 0:
                print('File %d: Processed %d / %d lines' % (file_i+1, line_i+1, len(lines)))

        with open(lids_path, 'w+') as ouf:
            for l in new_lines:
                ouf.write('%s\n' % l)

if __name__ == "__main__":
    main()
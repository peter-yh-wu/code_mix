import langid
import os

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    txt_dir = os.path.join(data_dir, 'txt')
    scripted_txt_path = os.path.join(txt_dir, 'script.txt')
    lids_dir = os.path.join(data_dir, 'lids')
    if not os.path.exists(lids_dir):
        os.makedirs(lids_dir)
    scripted_lids_path = os.path.join(lids_dir, 'script_lids.txt')

    with open(scripted_txt_path, 'r') as inf:
        script_lines = inf.readlines()

    new_lines = []
    for i, l in enumerate(script_lines):
        l = l.strip()
        l_list = l.split()

        lids = []
        for w in l_list[1:]:
            lid = langid.classify(w)[0]
            lids.append(lid)

        new_l_list = [l_list[0]]+lids
        new_l = ' '.join(new_l_list)
        new_lines.append(new_l)

        if (i+1) % 1000 == 0:
            print('Processed %d / %d lines' % (i+1, len(script_lines)))

    with open(scripted_lids_path, 'w+') as ouf:
        for l in new_lines:
            ouf.write('%s\n' % l)

if __name__ == "__main__":
    main()
import os

def process_dir(d, did, out_dir):
    files = os.listdir(d)
    files = [f for f in files if f.endswith('.TXT')]
    paths = [os.path.join(d, f) for f in files]
    for i, p in enumerate(paths):
        f = files[i]
        fid = did+'_'+f[:-4]
        with open(p, 'r') as inf:
            lines = inf.readlines()
        new_lines = []
        for l in lines:
            l = l.strip()
            if len(l) > 3 and l[0] != '#':
                l_list = l.split()
                t1 = float(l_list[0])
                t2 = float(l_list[1])
                text = ' '.join(l_list[3:])
                t1_str = str(t1).replace('.', 'p')
                t2_str = str(t2).replace('.', 'p')
                new_fid = fid+'_'+t1_str+'_'+t2_str
                new_l = new_fid + ' ' + text
                new_lines.append(new_l)

        new_p = os.path.join(out_dir, fid+'.txt')
        with open(new_p, 'w+') as ouf:
            for l in new_lines:
                ouf.write('%s\n' % l)

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    raw_text_dir = os.path.join(data_dir, 'raw_text')
    new_text_dir = os.path.join(data_dir, 'text')
    if not os.path.exists(new_text_dir):
        os.makedirs(new_text_dir)
    sub_raw_text_dirs = os.listdir(raw_text_dir)
    sub_raw_text_dirs = [d for d in sub_raw_text_dirs if len(d) == 3]
    sub_raw_text_dir_paths = [os.path.join(raw_text_dir, d) for d in sub_raw_text_dirs]

    for d, did in zip(sub_raw_text_dir_paths, sub_raw_text_dirs):
        process_dir(d, did, new_text_dir)

if __name__ == '__main__':
    main()
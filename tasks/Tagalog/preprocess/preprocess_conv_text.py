import os

def process_text(l):
    l_list = l.split()
    new_l_list = []
    for w in l_list:
        if w[0] != '<' and w != '(())':
            new_l_list.append(w)
    new_l = ' '.join(new_l_list)
    return new_l

def parse_txt(lines):
    new_lines = []
    curr_line_type = 'time' # or text
    start_timestamp = ''
    end_timestamp = ''
    text = ''
    for l in lines:
        l = l.strip()
        if curr_line_type == 'time':
            if start_timestamp == '': # only for first line
                start_timestamp = l[1:-1].replace('.', 'p')
            else:
                end_timestamp = l[1:-1].replace('.', 'p')
                if text != '' and text != '<no-speech>':
                    text = process_text(text)
                    if len(text) > 0:
                        fid = start_timestamp+'_'+end_timestamp
                        new_l = fid+' '+text
                        new_lines.append(new_l)
                start_timestamp = end_timestamp
            curr_line_type = 'text'
        elif curr_line_type == 'text':
            if l[0] == '[':
                start_timestamp = l[1:-1].replace('.', 'p')
            else:
                text = l
                curr_line_type = 'time'
    return new_lines

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    conv_dir = os.path.join(data_dir, 'conversational')
    conv_txt_dir = os.path.join(conv_dir, 'transcription')
    
    conv_txt_files = os.listdir(conv_txt_dir)
    conv_txt_files = [f for f in conv_txt_files if f.endswith('.txt')]
    conv_txt_paths = [os.path.join(conv_txt_dir, f) for f in conv_txt_files]

    new_txt_dir = os.path.join(data_dir, 'text')
    if not os.path.exists(new_txt_dir):
        os.makedirs(new_txt_dir)
    new_conv_txt_path = os.path.join(new_txt_dir, 'conv.txt')

    all_new_lines = []
    for i, p in enumerate(conv_txt_paths):
        f = conv_txt_files[i]
        fid = f[:-4]
        with open(p, 'r') as inf:
            lines = inf.readlines()
        new_lines = parse_txt(lines)
        new_lines = [fid+'_'+l for l in new_lines]
        all_new_lines += new_lines
    
    with open(new_conv_txt_path, 'w+') as ouf:
        for l in all_new_lines:
            ouf.write('%s\n' % l)

if __name__ == '__main__':
    main()
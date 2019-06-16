import os

from pydub import AudioSegment

def crop_wavs(orig_wav_path, new_wav_paths, t1s, t2s):
    audio = AudioSegment.from_file(orig_wav_path, "flac")
    for new_wav_path, t1, t2 in zip(new_wav_paths, t1s, t2s):
        newAudio = audio[t1:t2]
        newAudio.export(new_wav_path, format="wav")

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_dir = os.path.join(parent_dir, 'data', 'raw_wav')
    new_wav_dir = os.path.join(parent_dir, 'data', 'wav')
    raw_text_dir = os.path.join(parent_dir, 'data', 'raw_text')

    raw_text_files = os.listdir(raw_text_dir)
    raw_text_files = [f for f in raw_text_files if f.endswith('.txt')]
    raw_text_paths = [os.path.join(raw_text_dir, f) for f in raw_text_files]

    for i, raw_text_path in enumerate(raw_text_paths):
        curr_raw_text_file = raw_text_files[i]
        fid = curr_raw_text_file.split('_')[0]
        print('Processing %s' % fid)
        orig_wav_file = fid+'.wav'
        orig_wav_path = os.path.join(wav_dir, orig_wav_file)

        with open(raw_text_path, 'r') as inf:
            curr_lines = inf.readlines()
        t1s = []
        t2s = []
        new_wav_paths = []
        for l in curr_lines:
            l_list = l.split()
            t1_str = l_list[0]
            t2_str = l_list[1]
            t1 = float(t1_str)
            t2 = float(t2_str)
            t1s.append(t1)
            t2s.append(t2)
            t1_str = t1_str.replace('.', 'p')
            t2_str = t2_str.replace('.', 'p')
            new_wav_file = fid+'_'+t1_str+'_'+t2_str+'.wav'
            new_wav_path = os.path.join(new_wav_dir, new_wav_file)
            new_wav_paths.append(new_wav_path)
        crop_wavs(orig_wav_path, new_wav_paths, t1s, t2s)

if __name__ == '__main__':
    main()
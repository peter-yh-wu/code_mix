import os
from pydub import AudioSegment

def mk_wav_segments(text_path, raw_wav_path, out_dir):
    audio = AudioSegment.from_file(raw_wav_path, "wav")
    with open(text_path, 'r') as inf:
        lines = inf.readlines()
    for l in lines:
        l = l.strip()
        l_list = l.split()
        fid = l_list[0]
        t2_start_i = fid.rfind('_')+1
        t1_end_i = t2_start_i-1
        t1_start_i = fid.rfind('_', 0, t1_end_i)+1
        t1 = float(fid[t1_start_i:t1_end_i].replace('p', '.'))
        t2 = float(fid[t2_start_i:].replace('p', '.'))
        t1_ms = t1*1000
        t2_ms = t2*1000
        new_wav_path = os.path.join(out_dir, fid+'.wav')
        newAudio = audio[t1_ms:t2_ms]
        newAudio.export(new_wav_path, format="wav")

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    text_dir = os.path.join(data_dir, 'text')
    raw_wav_dir = os.path.join(data_dir, 'raw_wav')
    text_files = os.listdir(text_dir)
    text_files = [f for f in text_files if f.endswith('.txt')]
    text_paths = [os.path.join(text_dir, f) for f in text_files]
    wav_dir = os.path.join(data_dir, 'wav')
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    for i, text_path in enumerate(text_paths):
        text_file = text_files[i]
        fid = text_file[:-4]
        raw_wav_file = fid+'.wav'
        raw_wav_path = os.path.join(raw_wav_dir, raw_wav_file)
        mk_wav_segments(text_path, raw_wav_path, wav_dir)

if __name__ == '__main__':
    main()
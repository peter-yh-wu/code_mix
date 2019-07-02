import audioread
import contextlib
import os
import wave

def convert_sph_to_wav(in_path, out_path):
    with audioread.audio_open(in_path) as f:
        if f.channels == 0:
            print('no channels')
        else:
            with contextlib.closing(wave.open(out_path, 'w')) as of:
                of.setnchannels(f.channels)
                of.setframerate(f.samplerate)
                of.setsampwidth(2)
                for buf in f:
                    of.writeframes(buf)

def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')
    raw_speech_dir = os.path.join(data_dir, 'raw_speech')

    raw_wav_dir = os.path.join(data_dir, 'raw_wav')
    if not os.path.exists(raw_wav_dir):
        os.makedirs(raw_wav_dir)

    raw_speech_sub_dirs = os.listdir(raw_speech_dir)
    raw_speech_sub_dirs = [d for d in raw_speech_sub_dirs if len(d) == 13]
    raw_speech_sub_dirs = [os.path.join(raw_speech_dir, d) for d in raw_speech_sub_dirs]
    for raw_speech_sub_dir in raw_speech_sub_dirs:
        audio_dir = os.path.join(raw_speech_sub_dir, 'AUDIO')
        audio_sub_dirs = os.listdir(audio_dir)
        audio_sub_dirs = [d for d in audio_sub_dirs if len(d) == 3]
        audio_sub_dir_paths = [os.path.join(audio_dir, d) for d in audio_sub_dirs]
        for audio_sub_dir, audio_sub_dir_path in zip(audio_sub_dirs, audio_sub_dir_paths):
            raw_sph_files = os.listdir(audio_sub_dir_path)
            raw_sph_files = [f for f in raw_sph_files if f.endswith('SPH')]
            raw_sph_paths = [os.path.join(audio_sub_dir_path, f) for f in raw_sph_files]
            raw_wav_files = [audio_sub_dir+'_'+sph_file[:-4]+'.wav' for sph_file in raw_sph_files]
            raw_wav_paths = [os.path.join(raw_wav_dir, f) for f in raw_wav_files]
            for i, (sph_path, wav_path) in enumerate(zip(raw_sph_paths, raw_wav_paths)):
                print(raw_sph_files[i])
                convert_sph_to_wav(sph_path, wav_path)

if __name__ == '__main__':
    main()
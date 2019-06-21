import audioread
import contextlib
import os
import wave

def main():
  parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  data_dir = os.path.join(parent_dir, 'data')
  scripted_audio_dir = os.path.join(data_dir, 'scripted/audio')

  scripted_audio_files = os.listdir(scripted_audio_dir)
  scripted_audio_files = [f for f in scripted_audio_files if f.endswith('.sph')]
  scripted_audio_paths = [os.path.join(scripted_audio_dir, f) for f in scripted_audio_files]

  wav_dir = os.path.join(data_dir, 'wav')
  if not os.path.exists(wav_dir):
    os.makedirs(wav_dir)

  for i, p in enumerate(scripted_audio_paths):
    curr_audio_file = scripted_audio_files[i]
    curr_wav_file = curr_audio_file[:-4]+'.wav'
    curr_wav_path = os.path.join(wav_dir, curr_wav_file)

    with audioread.audio_open(p) as f:
      with contextlib.closing(wave.open(curr_wav_path, 'w+')) as of:
        of.setnchannels(f.channels)
        of.setframerate(f.samplerate)
        of.setsampwidth(2)

        for buf in f:
            of.writeframes(buf)

if __name__ == '__main__':
    main()
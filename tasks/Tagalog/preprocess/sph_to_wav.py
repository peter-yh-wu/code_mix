import audioread
import contextlib
import os
import wave

def main():
  parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  data_dir = os.path.join(parent_dir, 'data')
  audio_dir = os.path.join(data_dir, 'conversational/audio') # 'scripted/audio')

  audio_files = os.listdir(audio_dir)
  audio_files = [f for f in audio_files if f.endswith('.sph')]
  audio_paths = [os.path.join(audio_dir, f) for f in audio_files]

  wav_dir = os.path.join(data_dir, 'wav')
  if not os.path.exists(wav_dir):
    os.makedirs(wav_dir)

  for i, p in enumerate(audio_paths):
    curr_audio_file = audio_files[i]
    print(curr_audio_file)
    curr_wav_file = curr_audio_file[:-4]+'.wav'
    curr_wav_path = os.path.join(wav_dir, curr_wav_file)

    with audioread.audio_open(p) as f:
      with contextlib.closing(wave.open(curr_wav_path, 'w')) as of:
        of.setnchannels(f.channels)
        of.setframerate(f.samplerate)
        of.setsampwidth(2)

        for buf in f:
            of.writeframes(buf)

if __name__ == '__main__':
    main()
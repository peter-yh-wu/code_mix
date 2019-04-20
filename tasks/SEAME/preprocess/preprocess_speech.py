'''
Script to preprocess audio data

Peter Wu
peterw1@andrew.cmu.edu
'''

import os
from pydub import AudioSegment

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERVIEW_TEXT_DIR_I = os.path.join(parent_dir, 'data/interview/transcript/phaseI')
INTERVIEW_TEXT_DIR_II = os.path.join(parent_dir, 'data/interview/transcript/phaseII')
INTERVIEW_AUDIO_DIR = os.path.join(parent_dir, 'data/interview/audio')
INTERVIEW_WAV_DIR_I = os.path.join(parent_dir, 'data/interview/wavI')
INTERVIEW_WAV_DIR_II = os.path.join(parent_dir, 'data/interview/wavII')
CONVO_TEXT_DIR_I = os.path.join(parent_dir, 'data/conversation/transcript/phaseI')
CONVO_TEXT_DIR_II = os.path.join(parent_dir, 'data/conversation/transcript/phaseII')
CONVO_AUDIO_DIR = os.path.join(parent_dir, 'data/conversation/audio')
CONVO_WAV_DIR_I = os.path.join(parent_dir, 'data/conversation/wavI')
CONVO_WAV_DIR_II = os.path.join(parent_dir, 'data/conversation/wavII')

def crop_wav(in_path, out_path, t1, t2):
    '''t1 and t2 are in milliseconds, assumes both paths are .wav files'''
    newAudio = AudioSegment.from_file(in_path, "flac")
    newAudio = newAudio[t1:t2]
    newAudio.export(out_path, format="wav")

def crop_wavs(audio_path, wav_paths, t1s, t2s):
    audio = AudioSegment.from_file(audio_path, "flac")
    for wav_path, t1, t2 in zip(wav_paths, t1s, t2s):
        newAudio = audio[t1:t2]
        newAudio.export(wav_path, format="wav")

def crop_data(txt_dir, audio_dir, wav_dir):
    txt_files = os.listdir(txt_dir)
    txt_files = [f for f in txt_files if f.endswith('.txt')]
    txt_paths = [os.path.join(txt_dir, f) for f in txt_files]
    for f in txt_paths:
        with open(f, 'r') as inf:
            lines = inf.readlines()
        fid = lines[0].split()[0]
        audio_file = fid+'.flac'
        audio_path = os.path.join(audio_dir, audio_file)
        wav_paths = []
        t1s = []
        t2s = []
        for l in lines:
            tokens = l.split()
            t1 = int(tokens[1])
            t2 = int(tokens[2])
            wav_file = fid+'_'+tokens[1]+'_'+tokens[2]+'.wav'
            wav_path = os.path.join(wav_dir, wav_file)
            wav_paths.append(wav_path)
            t1s.append(t1)
            t2s.append(t2)
        crop_wavs(audio_path, wav_paths, t1s, t2s)

'''
if not os.path.exists(INTERVIEW_WAV_DIR_I):
    os.makedirs(INTERVIEW_WAV_DIR_I)
if not os.path.exists(CONVO_WAV_DIR_I):
    os.makedirs(CONVO_WAV_DIR_I)
crop_data(INTERVIEW_TEXT_DIR_I, INTERVIEW_AUDIO_DIR, INTERVIEW_WAV_DIR_I)
crop_data(CONVO_TEXT_DIR_I, CONVO_AUDIO_DIR, CONVO_WAV_DIR_I)
'''
if not os.path.exists(INTERVIEW_WAV_DIR_II):
    os.makedirs(INTERVIEW_WAV_DIR_II)
if not os.path.exists(CONVO_WAV_DIR_II):
    os.makedirs(CONVO_WAV_DIR_II)
crop_data(INTERVIEW_TEXT_DIR_II, INTERVIEW_AUDIO_DIR, INTERVIEW_WAV_DIR_II)
crop_data(CONVO_TEXT_DIR_II, CONVO_AUDIO_DIR, CONVO_WAV_DIR_II)
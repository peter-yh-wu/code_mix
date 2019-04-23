import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
interview_mfcc1_dir = os.path.join(parent_dir, 'data/interview/mfcc1')
interview_mfcc2_dir = os.path.join(parent_dir, 'data/interview/mfcc2')
conversation_mfcc1_dir = os.path.join(parent_dir, 'data/conversation/mfcc1')
conversation_mfcc2_dir = os.path.join(parent_dir, 'data/conversation/mfcc2')
interview_wav1_dir = os.path.join(parent_dir, 'data/interview/wavI')
interview_wav2_dir = os.path.join(parent_dir, 'data/interview/wavII')
conversation_wav1_dir = os.path.join(parent_dir, 'data/interview/wavI')
conversation_wav2_dir = os.path.join(parent_dir, 'data/interview/wavII')

interview_mfcc1_files = os.listdir(interview_mfcc1_dir)
interview_mfcc1_files = [f for f in interview_mfcc1_files if f.endswith('.mfcc')]
interview_mfcc2_files = os.listdir(interview_mfcc2_dir)
interview_mfcc2_files = [f for f in interview_mfcc2_files if f.endswith('.mfcc')]
conversation_mfcc1_files = os.listdir(conversation_mfcc1_dir)
conversation_mfcc1_files = [f for f in conversation_mfcc1_files if f.endswith('.mfcc')]
conversation_mfcc2_files = os.listdir(conversation_mfcc2_dir)
conversation_mfcc2_files = [f for f in conversation_mfcc2_files if f.endswith('.mfcc')]
interview_wav1_files = os.listdir(interview_wav1_dir)
interview_wav1_files = [f for f in interview_wav1_files if f.endswith('.wav')]
interview_wav2_files = os.listdir(interview_wav2_dir)
interview_wav2_files = [f for f in interview_wav2_files if f.endswith('.wav')]
conversation_wav1_files = os.listdir(conversation_wav1_dir)
conversation_wav1_files = [f for f in conversation_wav1_files if f.endswith('.wav')]
conversation_wav2_files = os.listdir(conversation_wav2_dir)
conversation_wav2_files = [f for f in conversation_wav2_files if f.endswith('.wav')]

interview_mfcc1_set = set([f[:-5] for f in interview_mfcc1_files])
interview_mfcc2_set = set([f[:-5] for f in interview_mfcc2_files])
conversation_mfcc1_set = set([f[:-5] for f in conversation_mfcc1_files])
conversation_mfcc2_set = set([f[:-5] for f in conversation_mfcc2_files])
interview_wav1_set = set([f[:-4] for f in interview_wav1_files])
interview_wav2_set = set([f[:-4] for f in interview_wav2_files])
conversation_wav1_set = set([f[:-4] for f in conversation_wav1_files])
conversation_wav2_set = set([f[:-4] for f in conversation_wav2_files])

missing_interview1_mfccs = interview_wav1_set - interview_mfcc1_set
missing_interview2_mfccs = interview_wav2_set - interview_mfcc2_set
missing_conversation1_mfccs = conversation_wav1_set - conversation_mfcc1_set
missing_conversation2_mfccs = conversation_wav2_set - conversation_mfcc2_set

print('interview1: missing %d files' % len(missing_interview1_mfccs))
if len(missing_interview1_mfccs) > 0:
    print('e.g. %s' % missing_interview1_mfccs[0])
print('interview2: missing %d files' % len(missing_interview2_mfccs))
if len(missing_interview2_mfccs) > 0:
    print('e.g. %s' % missing_interview2_mfccs[0])
print('conversation1: missing %d files' % len(missing_conversation1_mfccs))
if len(missing_conversation1_mfccs) > 0:
    print('e.g. %s' % missing_conversation1_mfccs[0])
print('conversation2: missing %d files' % len(missing_conversation2_mfccs))
if len(missing_conversation2_mfccs) > 0:
    print('e.g. %s' % missing_conversation2_mfccs[0])

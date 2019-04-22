#!/bin/bash

cd ..
mkdir -p data/interview/mfcc
cd data/interview/mfcc
wget http://tts.speech.cs.cmu.edu/rsk/misc_stuff/feats_seame_interview_07march2019.tar.gz
tar -xvzf feats_seame_interview_07march2019.tar.gz
mv feats_interview/cleaned ..
cd ..
rm -rf mfcc
mv cleaned mfcc1
cd ..
mkdir -p conversation/mfcc
cd conversation/mfcc
wget http://tts.speech.cs.cmu.edu/rsk/misc_stuff/feats_seame_conversation_07march2019.tar.gz 
tar -xvzf feats_seame_conversation_07march2019.tar.gz
mv feats_conversation/cleaned ..
cd ..
rm -rf mfcc
mv cleaned mfcc1
cd ..
wget http://tts.speech.cs.cmu.edu/rsk/misc_stuff/feats_wavII.tar.gz
tar -xvzf feats_wavII.tar.gz
rm feats_wavII.tar.gz
mv feats_conversation/cleaned/ conversation/
mv conversation/cleaned/ conversation/mfcc2
mv feats_interview/cleaned/ interview/
mv interview/cleaned/ interview/mfcc2
rm -rf feats_conversation/
rm -rf feats_interview/
cd ../preprocess
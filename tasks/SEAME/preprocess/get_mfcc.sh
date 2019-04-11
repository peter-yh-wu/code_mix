#!/bin/bash

cd ..
mkdir -p data/interview/mfcc
cd data/interview/mfcc
wget http://tts.speech.cs.cmu.edu/rsk/misc_stuff/feats_seame_interview_07march2019.tar.gz
tar -xvzf feats_seame_interview_07march2019.tar.gz
mv feats_interview/cleaned ..
cd ..
rm -rf mfcc
mv cleaned mfcc
cd ../../../preprocess
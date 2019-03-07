#!/bin/bash

KALDI_DIR="/home/srallaba/tools/kaldi/egs/seame/s5"

cd ../data
DATA_DIR=$(pwd)

for category in conversation interview
 do
  echo $category
  for file in ${category}/wav/*.wav
  do
     fname=$(basename "$file" .wav)
     echo $fname $file >> ${category}/wav.scp
  done

  cut -d ' ' -f 1 ${category}/wav.scp > speakers.${category}
  cut -d ' ' -f 1 ${category}/wav.scp > utterances.${category}
  paste -d' ' utterances.${category} speakers.${category} > ${category}/utt2spk
  cd $KALDI_DIR
  ./utils/utt2spk_to_spk2utt.pl $DATA_DIR/${category}/utt2spk > $DATA_DIR/${category}/spk2utt
  ./utils/fix_data_dir.sh $DATA_DIR/${category}

  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 100 --cmd "run.pl" $DATA_DIR/${category} $DATA_DIR/exp/mfcc $DATA_DIR/mfcc_${category}
  steps/compute_cmvn_stats.sh $DATA_DIR/${category} $DATA_DIR/exp/mfcc $DATA_DIR/mfcc_${category}/
  cd $DATA_DIR
 done

cd ../preprocess
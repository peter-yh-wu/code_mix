#!/bin/bash

KALDI_DIR="/home/srallaba/tools/kaldi/egs/seame/s5"

cd ../data

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
  .$KALDI_DIR/utils/utt2spk_to_spk2utt.pl ${category}/utt2spk > ${category}/spk2utt
  .$KALDI_DIR/utils/fix_data_dir.sh ${category}

  .$KALDI_DIR/steps/make_mfcc.sh --mfcc-config $KALDI_DIR/conf/mfcc.conf --nj 100 --cmd "run.pl" ${category} exp/mfcc mfcc_${category}
  .$KALDI_DIR/steps/compute_cmvn_stats.sh ${category} exp/mfcc mfcc_${category}/

 done

cd ../preprocess
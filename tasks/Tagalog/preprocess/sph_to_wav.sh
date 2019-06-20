cd ../data/scripted/audio
for a in *.sph; do
    echo $a
    y=${a%.sph}
    sox $a "../wav/${y}.wav"
done
cd ../../scripted/preprocess
cd ../data/scripted/audio
for %%a in (*.sph);
    echo $a 
    sox "%%~a" "../wav/%%~na.wav"
done
cd ../../scripted/preprocess
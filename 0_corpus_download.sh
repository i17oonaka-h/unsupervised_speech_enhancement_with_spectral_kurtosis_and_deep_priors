corpus_dir="./corpus"
mkdir -p $corpus_dir
cd $corpus_dir

# If lj-speech dataset is not downloaded, download it
if [ ! -d "LJSpeech-1.1" ]; then
    echo "Downloading LJSpeech-1.1 dataset..."
    curl -O https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    tar -xvf LJSpeech-1.1.tar.bz2
    rm LJSpeech-1.1.tar.bz2
    echo "LJSpeech-1.1 dataset downloaded."
else
    echo "LJSpeech-1.1 dataset already exists."
fi

# If DEMAND dataset is not downloaded, download it
download_list=("NFIELD" "TBUS" "PSTATION" "SPSQUARE" "PRESTO")
mkdir -p "DEMAND"
cd "DEMAND"
for dataset in ${download_list[@]}; do
    if [ 1 ]; then
        echo "Downloading $dataset dataset..."
        # curl -O https://zenodo.org/records/1227121/files/${dataset}_16k.zip
        # unzip ${dataset}_16k.zip
        cd $dataset # only use ch01.wav
        for file in *.wav; do
            if [ ! -f "ch01.wav" ]; then
                rm $file
            fi
        done
        cd ..
        # rm ${dataset}_16k.zip
        echo "$dataset dataset downloaded."
    else
        echo "$dataset dataset already exists."
    fi
done
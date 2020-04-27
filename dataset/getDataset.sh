#/bin/bash

echo " **** WARNING **** "
echo "Make sure you put this script in a storage with large free space"
echo "DO NOT run it in the repo"

read -p "Do you want to quit? (y/n): " quit_option

if [ "$quit_option" == "y" ]; then
  exit 0
fi

#label.csv should not exist in current directory, or should be unpopulated.
read -p "git clone ISIC archive? (y/n): " git_option

if [ "$git_option" == "y" ]; then
  git clone https://github.com/GalAvineri/ISIC-Archive-Downloader.git
fi

mkdir Data/Images
mkdir Data/Descriptions

read -p "how many processes should the downloader run on?: " num_proc

python3 ./ISIC-Archive-Downloader/download_archive.py --images-dir ./Data/Images --descs-dir ./Data/Descriptions --filter malignant --p $num_proc

python3 ./ISIC-Archive-Downloader/download_archive.py --images-dir ./Data/Images --descs-dir ./Data/Descriptions --filter benign --p $num_proc

python3 makecsv.py

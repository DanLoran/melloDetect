#/bin/bash

#label.csv should not exist in current directory, or should be unpopulated. 
git clone git@github.com:DanLoran/ISIC-Archive-Downloader.git
cd ISIC-Archive-Downloader
pip3 install requests
pip3 install Pillow
pip3 install tqdm

python3 download_archive.py --images-dir ../Data/Images --descs-dir ../Data/Descriptions --filter malignant --q 1

python3 download_archive.py --images-dir ../Data/Images --descs-dir ../Data/Descriptions --filter benign --q 1


cd ..
python3 makecsv.py
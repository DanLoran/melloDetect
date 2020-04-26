#!/bin/bash

FILES=../dataset/Data/Images/*
DEST=Data

mkdir $DEST

for f in $FILES
do
  echo "Processing $f file..."
  python3 augment.py $f $DEST/
done

echo "Generating labels..."
python3 augmented_labels.py ../dataset/label.csv $DEST/label.csv $DEST/

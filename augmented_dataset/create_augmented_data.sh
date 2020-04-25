#!/bin/bash

FILES=../dataset/Data/images/*
DEST=Data/

for f in $FILES
do
  echo "Processing $f file..."
  python3 augment.py $f Data/
done

echo "Generating labels..."
python3 augmented_labels.py ../dataset/label.csv ./label.csv Data/

import os
import argparse
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an augmented label set')
    parser.add_argument('source_labels_csv', type=str, help='source label csv path')
    parser.add_argument('dest_labels_csv', type=str, help='dest label csv path')
    parser.add_argument('images_dir', type=str, help='directory where the augmented images are stored')
    options = parser.parse_args()

    augmented_files = os.listdir(options.images_dir)
    basename_to_augments_map = {}
    for file in augmented_files:
        augmented_basename = os.path.splitext(file)[0]
        basename = augmented_basename[:augmented_basename.rfind('_')]
        if basename in basename_to_augments_map:
            basename_to_augments_map[basename].append(file)
        else:
            basename_to_augments_map[basename] = [file]

    print(basename_to_augments_map)

    # We assume the filenames are in the first column
    with open(options.source_labels_csv) as source_labels:
        with open(options.dest_labels_csv, 'w') as dest_labels:
            writer = csv.writer(dest_labels)
            reader = csv.reader(source_labels)
            for row in reader:
                if row[0] in basename_to_augments_map:
                    for file in basename_to_augments_map[row[0]]:
                        writer.writerow([file] + row[1:])

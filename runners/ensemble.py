import argparse
import csv
import os
import numpy as np

from mellolib.commonParser import LoadFromFile

def ensemble(dir, mode, output):

    n_files = len(os.listdir(dir))

    # check that at least two files are in the target directory
    assert n_files >= 2

    output_data = None

    if mode == "max":

        for f in os.listdir(dir):
            filename = os.path.join(dir, f)
            f_data = np.genfromtxt(filename, delimiter=',', usecols=(1))

            if output_data is None:
                output_data = np.reshape(f_data, (-1, 1))
                continue

            output_data = np.concatenate((output_data, np.reshape(f_data, (-1,1))), axis=1)

        indices = np.argmax(np.abs(output_data - 0.5), axis=1)

        t_data = np.empty(indices.shape)
        for r in range(indices.shape[0]):
            t_data[r] = output_data[r, indices[r]]

        output_data = t_data

    elif mode == "mean":

        for f in os.listdir(dir):
            filename = os.path.join(dir, f)
            f_data = np.genfromtxt(filename, delimiter=',', usecols=(1))

            if output_data is None:
                output_data = f_data / n_files


    f_strings = np.genfromtxt(filename, delimiter=',', usecols=(0), dtype=str)
    output_table = np.empty((f_strings.shape[0], 2))

    outputFile = open(output, "w")
    outputFile.write("image_name,target\n")
    for row in range(1, f_strings.shape[0]):
        outputFile.write(f_strings[row] + "," + str(output_data[row]) + "\n")
    outputFile.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=open, action=LoadFromFile)
    parser.add_argument("--dir", "-d", default=None,
               help="Directory where the predictions for all networks are located.")
    parser.add_argument("--mode", "-m", choices="['max', 'mean']", default="mean")
    parser.add_argument("--output", "-o", default=None,
               help="Output file path.")
    options = parser.parse_args()

    assert options.dir is not None

    if options.output is None:
        options.output = os.path.join(options.dir, "ensamble_output.csv")

    ensemble(options.dir, options.mode, options.output)

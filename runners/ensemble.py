import argparse
import csv
import os
import numpy as np

from mellolib.commonParser import LoadFromFile

def ensemble(dir, mode, output):

    n_files = len(os.listdir(dir))

    # check that at least two files are in the target directory
    assert n_files < 2

    output_data = None

    for f in os.listdir(dir):
        f_data = np.genfromtxt(f, delimiter=',', usecols=(1))

        if mode == "max":

            if output_data is None:
                output_data = np.abs(f_data - 0.5)
                continue

            output_data = np.maximum(np.abs(f_data - 0.5), output_data)

        elif mode == "mean":

            if output_data is None:
                output_data = f_data / n_files

    np.savetxt(output, output_data)

if __name__ == "__main__":

    parser = argparse.Argument()
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

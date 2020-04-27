import argparse
import csv
import shutil
from random import randint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float)
    parser.add_argument("--source-addr", type=str)
    parser.add_argument("--train-addr", type=str)
    parser.add_argument("--val-addr", type=str)
    options = parser.parse_args()

    with open("label.csv", "r") as f0:
        with open(options.val_addr+"/label.csv", "w") as f1:
            with open(options.train_addr+"/label.csv", "w") as f2:
                reader = csv.reader(f0)
                trainwriter = csv.writer(f1)
                valwriter = csv.writer(f2)
                for row in reader:
                    #Validation set
                    if(options.p >= randint(0,100)):
                        try:
                            valwriter.writerow(row)
                            shutil.copy(options.source_addr+'/'+row[0]+'.jpeg', options.val_addr+'/'+row[0]+'.jpeg')
                        except FileNotFoundError:
                            continue
                    else:
                        try:
                            trainwriter.writerow(row)
                            shutil.copy(options.source_addr+'/'+row[0]+'.jpeg', options.train_addr+'/'+row[0]+'.jpeg')
                        except FileNotFoundError:
                            continue

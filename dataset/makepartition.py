import argparse
import csv
import shutil
from random import randint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, help="percentage split")
    parser.add_argument("--source-addr", type=str)
    parser.add_argument("--train-addr", type=str)
    parser.add_argument("--val-addr", type=str)
    parser.add_argument("--max-pull", type=int, default = -1, help="total number of images pull from database")
    options = parser.parse_args()

    with open(options.source_addr+"/label.csv", "r") as f0:
        with open(options.val_addr+"/label.csv", "w") as f1:
            with open(options.train_addr+"/label.csv", "w") as f2:
                reader = csv.reader(f0)
                trainwriter = csv.writer(f1)
                valwriter = csv.writer(f2)
                count = 0
                for row in reader:
                    if (count > options.max_pull and options.max_pull != -1):
                        break
                    count+=1
                    print(count)
                    #Validation set
                    if(options.p >= randint(0,100)):
                        try:
                            shutil.copy(options.source_addr+'/'+row[0]+'.jpeg', options.val_addr+'/'+row[0]+'.jpeg')
                        except FileNotFoundError:
                            continue
                        valwriter.writerow(row)
                    else:
                        try:
                            shutil.copy(options.source_addr+'/'+row[0]+'.jpeg', options.train_addr+'/'+row[0]+'.jpeg')
                        except FileNotFoundError:
                            continue
                        trainwriter.writerow(row)

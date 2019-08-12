import os
import sys
import csv
import numpy as np
import random

from random import shuffle


def parse_file_name(file_name):
    person = int(file_name[1])
    gesture = int(file_name[3])
    take = int(file_name[5: 7])

    return person, gesture, take


def paths2csv(paths, f, replace=None):
    for path in paths:
        if replace is None:
            print(path, file=f)
        else:
            partial_path, file_name = os.path.split(path)
            file_name = file_name.replace(replace[0], replace[1])
            print(os.path.join(partial_path, file_name), file=f)

if __name__ == "__main__":
    random.seed(42)  # reproducing purposes
    walk_dir = os.getcwd()

    paths = [os.path.join(root, f)
             for root, dirs, files in os.walk(walk_dir)
             for f in files
             if "s" in f and "png" in f
             ]
    shuffle(paths)

    paths_train, paths_val = [], []
    used = []
    for path in paths:
        _, file_name = os.path.split(path)
        person, gesture, take = parse_file_name(file_name)

        if (person, gesture, take) not in used:
            used.append((person, gesture, take))
            paths_val.append(path)
        else:
            paths_train.append(path)

    print(f"{len(paths_train)} frames for training")
    print(f"{len(paths_val)} frames for validation")

    # (!) will overwrite
    with open("train_data2.csv", "w") as f:
        paths2csv(paths_train, f, ("s", "d"))  # d stands for depth
    with open("val_data2.csv", "w") as f:
        paths2csv(paths_val, f, ("s", "d"))  # d stands for depth
    with open("train_labels2.csv", "w") as f:
        paths2csv(paths_train, f, ("s", "w"))  # w stands for sparsed
    with open("val_labels2.csv", "w") as f:
        paths2csv(paths_val, f, ("s", "w"))  # w stands for sparsed

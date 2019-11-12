import os
import random
from train_test_split import random_k_items

ROOT_DIR = '../aishell_2_partitions'
CLASS_TRAIN_FILE = f'{ROOT_DIR}/class_train.txt'
SELECTION_SEED = 123
SELECTION_RATIO = 0.2
DATA_ROOT_DIR = '../aishell_2/data/wav/augmented'

# output file
CLASS_TRAIN_AUG = f'{ROOT_DIR}/class_train_aug.txt'

def random_select_and_write(input_file, output_file, ratio, seed):
    total_list = []
    with open(input_file, 'r') as infile:
        for line in infile:
            wav, label = line.strip().split()
            total_list.append((wav, label))
    
    total_len = len(total_list)
    selected_len = int(total_len * ratio)
    selected_list = random_k_items(total_list, selected_len, seed)
    with open(output_file, 'w') as outfile:
        for wav, label in selected_list:
            outfile.writelines(f'{wav}\t{label}\n')

random_select_and_write(CLASS_TRAIN_FILE, CLASS_TRAIN_AUG, SELECTION_RATIO, SELECTION_SEED)
import os
import random
from train_test_split import random_k_items

ROOT_DIR = '../aishell_2_partitions'
CLASS_TRAIN_FILE = f'{ROOT_DIR}/class_train.txt'
ENC_TRAIN_FILE = f'{ROOT_DIR}/enc_train.txt'
CLASS_SELECTION_SEED = 123
ENC_SELECTION_SEED = 321
SELECTION_RATIO = 0.2
DATA_ROOT_DIR = '../aishell_2/data/wav/augmented'

# output files
CLASS_TRAIN_AUG = f'{ROOT_DIR}/class_train_aug.txt'
ENC_TRAIN_AUG = f'{ROOT_DIR}/enc_train_aug.txt'

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
            _, _, _, _, _, wav_name = wav.split('/')
            new_wav = f'{DATA_ROOT_DIR}/{wav_name[:-4]}_aug.wav'
            outfile.writelines(f'{new_wav}\t{label}\n')

random_select_and_write(ENC_TRAIN_FILE, ENC_TRAIN_AUG, SELECTION_RATIO, ENC_SELECTION_SEED)
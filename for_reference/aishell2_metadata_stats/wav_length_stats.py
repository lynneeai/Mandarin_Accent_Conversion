import os
import random
import parselmouth
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = '../aishell_2_partitions'

def random_k_items(l, k, seed):
	list_length = len(l)
	if k > list_length:
		raise Exception('k should be smaller than list length')
	idx = set()
	random.seed(seed)
	while len(idx) < k:
		idx.add(random.randint(0, list_length - 1))
	return [l[i] for i in idx]


total_lines = []
with open(f'{ROOT_DIR}/class_train.txt', 'r') as infile:
	counter = 1
	for line in infile:
		total_lines.append(line)

selected_lines = random_k_items(total_lines, 5000, 123)
time_list = []
for line in tqdm(selected_lines):
	wav_file, _ = line.split()
	sound = parselmouth.Sound(f'{wav_file}')
	time = float(sound.get_total_duration())
	time_list.append(time)

ax = sns.distplot(time_list)
plt.tight_layout()
plt.savefig('wav_length_dist.png')

time_list = np.array(time_list)
with open('wav_length_stats.txt', 'w') as outfile:
	outfile.writelines(f'mean value: {np.mean(time_list)}\n')
	outfile.writelines(f'min value: {np.min(time_list)}\n')
	outfile.writelines(f'max value: {np.max(time_list)}\n')
	outfile.writelines(f'80 percentile: {np.percentile(time_list, 90)}\n')
	outfile.writelines(f'90 percentile: {np.percentile(time_list, 90)}\n')
import os
import parselmouth
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = '../aishell_2/data'

time_list = []
with open(f'{ROOT_DIR}/wav.scp', 'r') as infile:
	for line in infile:
		_, wav_file = line.split()
		sound = parselmouth.Sound(f'{ROOT_DIR}/{wav_file}')
		time = float(sound.get_total_duration())
		time_list.append(time)

ax = sns.distplot(time_list)
plt.tight_layout()
plt.show()

time_list = np.array(time_list)
with open('wav_length_stats.txt', 'w') as outfile:
	outfile.writelines(f'mean value: {np.mean(time_list)}')
	outfile.writelines(f'min value: {np.min(time_list)}')
	outfile.writelines(f'max value: {np.max(time_list)}')
	outfile.writelines(f'80 percentile: {np.percentile(time_list, 90)}\n')
	outfile.writelines(f'90 percentile: {np.percentile(time_list, 90)}\n')
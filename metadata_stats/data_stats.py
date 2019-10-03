import os
import argparse
import parselmouth
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from collections import defaultdict
from tqdm import tqdm

train_wavs = '../magicdata/metadata/train.scp'
dev_wavs = '../magicdata/metadata/dev.scp'
test_wavs = '../magicdata/metadata/test.scp'
spkinfo = '../magicdata/metadata/SPKINFO.txt'

train_folder = '../magicdata/train/'
dev_folder = '../magicdata/dev/'
test_folder = '../magicdata/test/'

spk_prov_dict = defaultdict()
with open(spkinfo, 'r') as spkfile:
	firstline = True
	for line in spkfile:
		if not firstline:
			info = line.strip().split('\t')
			spk_prov_dict[info[0]] = info[-1]
		firstline = False

def get_batch_spk_ids(batch_file):
	spk_ids = set()
	with open(batch_file, 'r') as infile:
		for line in infile:
			line = line.strip().split('\t')
			info = line[0].split('_')
			spk = '_'.join(info[:-1])
			spk_ids.add(spk)
	return spk_ids

def spk_prov_dist_plot(spk_ids, output_file):
	prov_count = defaultdict(int)
	for spk in spk_ids:
		prov_count[spk_prov_dict[spk]] += 1

	sorted_prov_count = sorted(list(prov_count.items()), key=lambda x: x[1], reverse=True)

	prov_df = pd.DataFrame(sorted_prov_count)
	prov_df.columns = ['Province', 'Count']

	ax = sns.barplot(x='Count', y='Province', data=prov_df)
	ax.set_yticklabels(prov_df.Province)
	for i, v in enumerate(prov_df['Count'].items()):        
		ax.text(v[1] * 1.01, i, f'{v[1]}', color='m')
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()

def get_batch_spk_time_dict(batch_spk_ids, batch_folder):
	print('----Processing wav files for each speakr...----\n')
	spk_time_dict = defaultdict(float)
	for id in tqdm(batch_spk_ids):
		for wav in os.listdir(f'{batch_folder}/{id}'):
			if wav.endswith('.wav'):
				sound = parselmouth.Sound(f'{batch_folder}/{id}/{wav}')
				time = float(sound.get_total_duration())
				spk_time_dict[id] += time
	print('----Finished processing wav files!----\n')
	return spk_time_dict

def time_prov_dist_plot(spk_time_dict, output_file):
	print('----Plotting...----\n')
	prov_time = defaultdict(float)
	for spk, time in spk_time_dict.items():
		prov = spk_prov_dict[spk]
		prov_time[prov] += time
	prov_time = {k: v / 3600 for k, v in prov_time.items()}
	
	sorted_prov_time = sorted(list(prov_time.items()), key=lambda x: x[1], reverse=True)

	prov_df = pd.DataFrame(sorted_prov_time)
	prov_df.columns = ['Province', 'Time']

	ax = sns.barplot(x='Time', y='Province', data=prov_df)
	ax.set_yticklabels(prov_df.Province)
	for i, v in enumerate(prov_df['Time'].items()):        
		ax.text(v[1] * 1.01, i, f'{v[1]}', color='m')
	plt.tight_layout()
	plt.savefig(output_file)
	plt.close()
	print('----Done!----\n')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str, choices=['get_spk_prov_dist', 'get_time_prov_dist'])
	args = parser.parse_args()

	if args.task == 'get_spk_prov_dist':
		for batch_file in [train_wavs, dev_wavs, test_wavs]:
			batch_name = batch_file.split('/')[-1][:-4]
			print(f'----Processing {batch_name} batch...----\n')
			ids = get_batch_spk_ids(batch_file)
			spk_prov_dist_plot(ids, f'spk_prov_{batch_name}.png')

	if args.task == 'get_time_prov_dist':
		for batch_file, batch_folder in [(train_wavs, train_folder), (dev_wavs, dev_folder), (test_wavs, test_folder)]:
			batch_name = batch_file.split('/')[-1][:-4]
			print(f'----Processing {batch_name} batch...----\n')
			ids = get_batch_spk_ids(batch_file)
			spk_time_dict = get_batch_spk_time_dict(ids, batch_folder)
			time_prov_dist_plot(spk_time_dict, f'time_prov_{batch_name}.png')
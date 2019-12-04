import os
import random
from collections import defaultdict

SUBSET_PERCENT = 0.1
SEED = 233
PAIRS = 8

def random_k_items(l, k, seed=None):
	list_length = len(l)
	if k > list_length:
		raise Exception('k should be smaller than list length')
	idx = set()
	if seed != None: random.seed(seed)
	while len(idx) < k:
		idx.add(random.randint(0, list_length - 1))
	return [l[i] for i in idx]

def complement_list(l_full, l_given):
	l_given_set = set(l_given)
	return [i for i in l_full if i not in l_given_set]

all_wavs = []
spk_dict = defaultdict(list)
with open('../aishell_2_partitions/class_test.txt', 'r') as infile:
	for line in infile:
		wav, label = line.split()
		spk = wav.split('/')[4]
		new_wav = f'../../{wav}'
		all_wavs.append(new_wav)
		spk_dict[spk].append(new_wav)

selected_wavs = random_k_items(all_wavs, int(len(all_wavs)*SUBSET_PERCENT), SEED)

with open('../accent_recognition/data/test/spk_trial', 'w') as outfile:
	for wav in selected_wavs:
		for i in range(PAIRS):
			spk = wav.split('/')[6]
			if i % 2 == 0:
				# select a different wav from the same speaker
				pair_wav = random_k_items(spk_dict[spk], 1)[0]
				while pair_wav == wav:
					pair_wav = random_k_items(spk_dict[spk], 1)[0]
				outfile.writelines(f'1 {wav} {pair_wav}\n')
			else:
				# select a different wav from a different speaker
				pair_wav = random_k_items(all_wavs, 1)[0]
				while pair_wav.split('/')[6] == spk:
					pair_wav = random_k_items(all_wavs, 1)[0]
				outfile.writelines(f'0 {wav} {pair_wav}\n')

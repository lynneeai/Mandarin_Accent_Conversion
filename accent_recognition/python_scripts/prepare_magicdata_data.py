# This file is created by Lin Ai (la2734@columbia.edu)

import os
import argparse
import random
from collections import defaultdict

MAGICDATA_ROOT = f'../magicdata'
DATA_ROOT = f'./data'
RANDOM_SEED = 23333
SHUFFLE_SEED = 32222
TEST_RATIO = 0.2

DIALECT_LABEL_DICT = {'bei jing':'guan', 'tian jin':'guan', 'he bei':'guan',
						'zhe jiang':'wu', 'shang hai':'wu', 'jiang su':'wu',
						'guang dong':'yue', 'guang xi':'yue',
						'chong qing':'chuan', 'si chuan':'chuan',
						'ji lin':'dongbei', 'liao ning':'dongbei', 'hei long jiang':'dongbei'}

LABEL_DIALECT_DICT = {'guan':['bei jing', 'tian jin', 'he bei'],
						'wu':['zhe jiang', 'shang hai', 'jiang su'],
						'yue':['guang dong', 'guang xi'],
						'chuan':['chong qing', 'si chuan'],
						'dongbei':['ji lin', 'liao ning', 'hei long jiang']}

SPK_PROV_DICT = defaultdict()
with open(f'{MAGICDATA_ROOT}/metadata/SPKINFO.txt', 'r') as infile:
	first_line = True
	for line in infile:
		if not first_line:
			spk_id, _, _, dialect = line.strip().split('\t')
			if dialect in DIALECT_LABEL_DICT:
				SPK_PROV_DICT[spk_id] = dialect
		first_line = False

def random_k_items(l, k, seed=None):
	list_length = len(l)
	if k == list_length: return l
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

def generate_group_wav_dict(batch_file):
	prov_wav_dict = defaultdict(list)
	wav_id_dict = defaultdict()
	with open(batch_file, 'r') as infile:
		for line in infile:
			wavname, wavdir = line.split()
			wav_id = wavname[:-4]
			wav_loc = f'{MAGICDATA_ROOT}/{wavdir}'
			spk_id = '_'.join(wavname.split('_')[:-1])
			if spk_id in SPK_PROV_DICT:
				wav_id_dict[wav_id] = wav_loc
				prov_wav_dict[SPK_PROV_DICT[spk_id]].append(wav_id)
	
	min_group_count = float('inf')
	min_group = ''
	for group, provs in LABEL_DIALECT_DICT.items():
		group_sum = 0
		for prov in provs:
			group_sum += len(prov_wav_dict[prov])
		if group_sum < min_group_count:
			min_group_count = group_sum
			min_group = group
	print(f'minimum length group: {min_group} {min_group_count}')

	group_wav_dict = defaultdict(list)
	for group in ['wu', 'yue', 'chuan', 'dongbei']:
		group_wavs = []
		for prov in LABEL_DIALECT_DICT[group]:
			group_wavs.extend(prov_wav_dict[prov])
		random.seed(SHUFFLE_SEED)
		random.shuffle(group_wavs)
		group_selected_wavs = random_k_items(group_wavs, min_group_count, RANDOM_SEED)
		group_wav_dict[group] = group_selected_wavs

	# process guan separatly to include as much beijing&tianjin as possible
	hebei_quota = min_group_count - len(prov_wav_dict['bei jing']) - len(prov_wav_dict['tian jin'])
	selected_hubei = random_k_items(prov_wav_dict['he bei'], hebei_quota, RANDOM_SEED)
	guan_selected_wavs = prov_wav_dict['bei jing'] + prov_wav_dict['tian jin'] + selected_hubei
	random.seed(SHUFFLE_SEED)
	random.shuffle(guan_selected_wavs)
	group_wav_dict['guan'] = guan_selected_wavs

	return group_wav_dict, wav_id_dict

def generate_wavscp_utt2spk(batch_dict, wav_id_dict, batchname):
	root = f'{DATA_ROOT}/{batchname}'
	if not os.path.exists(root):
		os.mkdir(root)
	wavscp_file = f'{root}/wav.scp'
	utt2spk_file = f'{root}/utt2spk'
	wavscp_writer = open(wavscp_file, 'w')
	utt2spk_writer = open(utt2spk_file, 'w')

	wavscp_list = []
	utt2spk_list = []
	for group, wav_list in batch_dict.items():
		for wav_id in wav_list:
			new_wav_id = f'{group}_{wav_id}'
			wavscp_list.append(f'{new_wav_id}\t{wav_id_dict[wav_id]}\n')
			utt2spk_list.append(f'{new_wav_id}\t{group}\n')
	
	wavscp_list.sort()
	utt2spk_list.sort()

	for item in wavscp_list:
		wavscp_writer.writelines(f'{item}')
	for item in utt2spk_list:
		utt2spk_writer.writelines(f'{item}')

	wavscp_writer.close()
	utt2spk_writer.close()

group_wav_dict, wav_id_dict = generate_group_wav_dict(f'{MAGICDATA_ROOT}/metadata/train.scp')
train_dict = defaultdict()
test_dict = defaultdict()
for group, wav_list in group_wav_dict.items():
	test_size = int(len(wav_list) * TEST_RATIO)
	test_list = random_k_items(wav_list, test_size, RANDOM_SEED)
	train_list = complement_list(wav_list, test_list)
	train_dict[group] = train_list
	test_dict[group] = test_list

generate_wavscp_utt2spk(train_dict, wav_id_dict, 'train_magicdata')
generate_wavscp_utt2spk(test_dict, wav_id_dict, 'test_magicdata')

import os
import random
from collections import defaultdict

SCP_FILE = '../accent_recognition/data/test_magicdata/wav.scp'
TRIAL_FILE = '../accent_recognition/data/test_magicdata/trials'
SUBSET_PERCENT = 0.2
SEED = 233333
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

all_utts = []
acc_dict = defaultdict(list)
with open(SCP_FILE, 'r') as infile:
    for line in infile:
        utt_id, wav = line.split()
        acc = utt_id.split('_')[0]
        all_utts.append(utt_id)
        acc_dict[acc].append(utt_id)
random.shuffle(all_utts)

selected_acc_dict = defaultdict(list)
for acc, utts in acc_dict.items():
    selected_utts = random_k_items(utts, int(len(utts)*SUBSET_PERCENT), SEED)
    selected_acc_dict[acc] = selected_utts

outfile = open(TRIAL_FILE, 'w')
for acc, utts in selected_acc_dict.items():
    for utt in utts:
        for i in range(PAIRS):
            if i % 2 == 0:
                # select a different utt from the same acc
                pair_utt = random_k_items(acc_dict[acc], 1)[0]
                while pair_utt == utt:
                    pair_utt = random_k_items(acc_dict[acc], 1)[0]
                outfile.writelines(f'{utt} {pair_utt} target\n')
            else:
                # select a different utt from a different acc
                pair_utt = random_k_items(all_utts, 1)[0]
                while pair_utt.split('_')[0] == acc:
                    pair_utt = random_k_items(all_utts, 1)[0]
                outfile.writelines(f'{utt} {pair_utt} nontarget\n')
outfile.close()
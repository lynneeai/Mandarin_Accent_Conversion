# This file is created by Lin Ai (la2734@columbia)
# author: Lin Ai (la2734)

import os
import librosa
import pickle
import h5py
import random
import multiprocessing as mp
import pyworld as pw
import numpy as np
from tqdm import tqdm
from collections import defaultdict

SR = 16000
FFT_SIZE = 256
FRAME_PERIOD = 10
DATA_ROOT = '../magicdata_data'
SCP_FILE = '../accent_recognition/data/train_magicdata/wav.scp'
PROCESS_NUM = 5
SUBSET_PERCENT = 0.2
BATCH_PERCENT = 0.05
SEED = 233

def random_k_items(l, k, seed):
	list_length = len(l)
	if k > list_length:
		raise Exception('k should be smaller than list length')
	idx = set()
	random.seed(seed)
	while len(idx) < k:
		idx.add(random.randint(0, list_length - 1))
	return [l[i] for i in idx]

def complement_list(l_full, l_given):
	l_given_set = set(l_given)
	return [i for i in l_full if i not in l_given_set]

def wav2pw(wavfile, sr=SR, fft_size=FFT_SIZE, frame_period=FRAME_PERIOD):
	x, _ = librosa.load(wavfile, sr=sr, mono=True, dtype=np.float64)
	_f0, t = pw.harvest(x, sr, frame_period=frame_period)
	f0 = pw.stonemask(x, _f0, t, sr)
	sp = pw.cheaptrick(x, f0, t, sr, fft_size=fft_size)
	ap = pw.d4c(x, f0, t, sr, fft_size=fft_size)
	return f0, sp, ap

def pw2wav(f0, sp, ap, sr=SR, fft_size=FFT_SIZE, frame_period=FRAME_PERIOD):
	wav = pw.synthesize(f0, sp, ap, SR, frame_period=frame_period)
	wav[np.where(np.isnan(wav))] = 0.0
	return wav

def extract_one_wav(wav_item):
	utt_id, label, wav_file = wav_item
	f0, sp, ap = wav2pw(wav_file)
	return utt_id, f0, sp, ap, label

def extract_sp(wav_list):
	total_wavs = len(wav_list)
	batch_size = int(total_wavs * BATCH_PERCENT)
	mini_batches = [wav_list[i*batch_size:(i+1)*batch_size] for i in range((total_wavs + batch_size - 1) // batch_size)]
	
	pool = mp.Pool(processes=PROCESS_NUM)
	i = 0
	for batch in mini_batches:
		print(f'processing mini batch {i}...')
		print(f'multiprocessing...')
		results = [pool.apply_async(extract_one_wav, args=(wav_item, )) for wav_item in tqdm(batch)]

		print(f'aggregating results...')
		output = [p.get() for p in tqdm(results)]
		
		print(f'writing mini batch {i} to files...')
		write_data(output, i)

		print(f'finished batch {i}!')
		i += 1

	pool.close()
	pool.join()

def write_data(object_list, batch_num):
	utt_id_list = []
	f0_list = []
	sp_list = []
	ap_list = []
	label_list = []

	for utt_id, f0, sp, ap, label in object_list:
		utt_id_list.append(utt_id)
		f0_list.append(np.asarray(f0))
		sp_list.append(np.asarray(sp))
		ap_list.append(np.asarray(ap))
		label_list.append(label)
	
	if not os.path.exists(DATA_ROOT):
		os.mkdir(DATA_ROOT)

	# write f0, sp, ap
	print('writing f0, sp, ap...')
	np.savez(f'{DATA_ROOT}/pw_{batch_num}', f0=f0_list, sp=sp_list, ap=sp_list)
	
	# write utt, label pickle
	print('writing utt_label...')
	utt_label = [utt_id_list, label_list]
	outfile = open(f'{DATA_ROOT}/utt_label_{batch_num}.pkl', 'wb') 
	pickle.dump(utt_label, outfile)
	outfile.close()

if __name__ == "__main__":
	acc_dict = defaultdict(list)
	with open(SCP_FILE, 'r') as infile:
		for line in infile:
			utt_id, wav = line.split()
			acc = utt_id.split('_')[0]
			wav_dir = wav[4:]
			acc_dict[acc].append((utt_id, acc, wav_dir))

	selected_acc_dict = defaultdict(list)
	for acc, utts in acc_dict.items():
		selected_utts = random_k_items(utts, int(len(utts)*SUBSET_PERCENT), SEED)
		selected_acc_dict[acc] = selected_utts

	selected_wavs = [wav for _, wav_list in selected_acc_dict.items() for wav in wav_list]
	extract_sp(selected_wavs)
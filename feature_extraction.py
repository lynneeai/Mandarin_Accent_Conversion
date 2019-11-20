import os
import librosa
import pickle
import h5py
import random
import multiprocessing as mp
import soundfile as sf
import pyworld as pw
import numpy as np
from tqdm import tqdm

SR = 16000
FFT_SIZE = 256
FRAME_PERIOD = 10
DATA_ROOT = './data'
PARTITION_FILE_ROOT = './aishell_2_partitions'
PROCESS_NUM = 10
BATCH_PERCENT = 1
SUBSET_PERCENT = 1
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

def extract_one_line(line):
	wav_file, label = line.split()
	new_wav_file = wav_file[1:]
	utt_id = wav_file.split('/')[-1][:-4]
	f0, sp, ap = wav2pw(new_wav_file)

	return utt_id, f0, sp, ap, label

def extract_data(partition_file, partition_name):
	print(f'extracting {partition_name}...')
	infile_lines = []
	with open(partition_file, 'r') as infile:
		for line in infile:
			infile_lines.append(line)
	infile_lines = random_k_items(infile_lines, int(len(infile_lines)*SUBSET_PERCENT), SEED)

	total_lines = len(infile_lines)
	batch_size = int(total_lines * BATCH_PERCENT)
	mini_batches = [infile_lines[i*batch_size:(i+1)*batch_size] for i in range((total_lines + batch_size - 1) // batch_size )]

	pool = mp.Pool(processes=PROCESS_NUM)
	i = 0
	for batch in mini_batches:
		print(f'processing mini batch {i}...')
		print(f'multiprocessing...')
		results = [pool.apply_async(extract_one_line, args=(line, )) for line in tqdm(batch)]

		print(f'aggregating results...')
		output = [p.get() for p in tqdm(results)]
		
		print(f'writing mini batch {i} to files...')
		write_data(output, partition_name, i)

		print(f'finished batch {i}!')
		i += 1

	pool.close()
	pool.join()

def write_data(object_list, partition_name, batch_num):
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
	
	partition_dir = f'{DATA_ROOT}/{partition_name}'
	if not os.path.exists(partition_dir):
		os.mkdir(partition_dir)

	# write f0, sp, ap
	print('writing f0, sp, ap...')
	np.savez(f'{partition_dir}/pw_{batch_num}', f0=f0_list, sp=sp_list, ap=sp_list)
	
	# write utt, label pickle
	print('writing utt_label...')
	utt_label = [utt_id_list, label_list]
	outfile = open(f'{partition_dir}/utt_label_{batch_num}.pkl', 'wb') 
	pickle.dump(utt_label, outfile)
	outfile.close()

def load_data(partition_name, batch_num):
	partition_dir = f'{DATA_ROOT}/{partition_name}'
	# load f0, sp, ap
	pw_list = np.load(f'{partition_dir}/pw_{batch_num}.npy', allow_pickle=True)

	# load utt, label
	utt_label_file = open(f'{partition_dir}/utt_label_{batch_num}.pkl', 'rb')
	utt_id_list, label_list = pickle.load(utt_label_file)
	utt_label_file.close()
	return utt_id_list, pw_list['f0'], pw_list['sp'], pw_list['ap'], label_list

if __name__ == "__main__":
	if not os.path.exists(DATA_ROOT):
		os.mkdir(DATA_ROOT)

	extract_data(f'{PARTITION_FILE_ROOT}/class_train.txt', 'class_train')
	extract_data(f'{PARTITION_FILE_ROOT}/class_dev.txt', 'class_dev')
	extract_data(f'{PARTITION_FILE_ROOT}/class_test.txt', 'class_test')

	extract_data(f'{PARTITION_FILE_ROOT}/enc_train.txt', 'enc_train')
	# extract_data(f'{PARTITION_FILE_ROOT}/enc_dev.txt', 'enc_dev')
	extract_data(f'{PARTITION_FILE_ROOT}/enc_test.txt', 'enc_test')

	extract_data(f'{PARTITION_FILE_ROOT}/class_train_aug_complete.txt', 'class_train_aug')
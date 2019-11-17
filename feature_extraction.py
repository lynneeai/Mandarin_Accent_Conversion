import os
import librosa
import pickle
import soundfile as sf
import pyworld as pw
import numpy as np
from tqdm import tqdm

SR = 16000
FFT_SIZE = 1024
DATA_ROOT = './data'
PARTITION_FILE_ROOT = './aishell_2_partitions'

def wav2pw(wavfile, sr=SR, fft_size=FFT_SIZE):
	x, _ = librosa.load(wavfile, sr=sr, mono=True, dtype=np.float64)
	_f0, t = pw.harvest(x, sr)
	f0 = pw.stonemask(x, _f0, t, sr)
	sp = pw.cheaptrick(x, f0, t, sr, fft_size=fft_size)
	ap = pw.d4c(x, f0, t, sr, fft_size=fft_size)
	return f0, sp, ap

def pw2wav(f0, sp, ap, sr=SR, fft_size=FFT_SIZE):
	wav = pw.synthesize(f0, sp, ap, SR)
	wav[np.where(np.isnan(wav))] = 0.0
	return wav

def extract_data(partition_file, partition_name):
	utt_id_list = []
	f0_list = []
	sp_list = []
	ap_list = []
	label_list = []

	print(f'extracting {partition_name}...')
	infile_lines = []
	with open(partition_file, 'r') as infile:
		for line in infile:
			infile_lines.append(line)

	for line in tqdm(infile_lines):
		wav_file, label = line.split()
		utt_id = wav_file.split('/')[5][:-4]
		wav_file = wav_file[3:]
		f0, sp, ap = wav2pw(wav_file)
		utt_id_list.append(utt_id)
		f0_list.append(f0)
		sp_list.append(sp)
		ap_list.append(ap)
		label_list.append(label)

	partition_dir = f'{DATA_ROOT}/{partition_name}'
	if not os.path.exists(partition_dir):
		os.mkdir(partition_dir)

	# write f0, sp, ap to numpy
	np.save(f'{partition_dir}/f0', np.array(f0_list))
	np.save(f'{partition_dir}/sp', np.array(sp_list))
	np.save(f'{partition_dir}/ap', np.array(ap_list))
	
	# write utt, label pickle
	utt_label = [utt_id_list, label_list]
	outfile = open(f'{partition_dir}/utt_label.pkl', 'wb') 
	pickle.dump(utt_label, outfile)
	outfile.close()

def load_data(partition_name):
	partition_dir = f'{DATA_ROOT}/{partition_name}'
	# load f0, sp, ap
	f0_list = np.load(f'{partition_dir}/f0.npy', allow_pickle=True)
	sp_list = np.load(f'{partition_dir}/sp.npy', allow_pickle=True)
	ap_list = np.load(f'{partition_dir}/ap.npy', allow_pickle=True)

	# load utt, label
	utt_label_file = open(f'{partition_dir}/utt_label.pkl', 'rb')
	utt_id_list, label_list = pickle.load(utt_label_file)
	utt_label_file.close()
	return utt_id_list, f0_list, sp_list, ap_list, label_list

if __name__ == "__main__":
	if not os.path.exists(DATA_ROOT):
		os.mkdir(DATA_ROOT)

	extract_data(f'{PARTITION_FILE_ROOT}/class_train.txt', 'class_train')
	extract_data(f'{PARTITION_FILE_ROOT}/class_train_aug_complete.txt', 'class_train_aug')
	extract_data(f'{PARTITION_FILE_ROOT}/class_dev.txt', 'class_dev')
	extract_data(f'{PARTITION_FILE_ROOT}/class_test.txt', 'class_test')
	
	extract_data(f'{PARTITION_FILE_ROOT}/enc_train.txt', 'enc_train')
	extract_data(f'{PARTITION_FILE_ROOT}/enc_dev.txt', 'enc_dev')
	extract_data(f'{PARTITION_FILE_ROOT}/enc_test.txt', 'enc_test')
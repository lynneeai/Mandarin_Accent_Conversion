# This file is created by Lin Ai (la2734@columbia.edu)

import os
import argparse

def get_wav_scp(batch_file, output_dir):
	wav_scp_list = []
	with open(batch_file, 'r') as infile:
		for line in infile:
			wav, _ = line.split()
			wav_id = wav.split('/')[-1][:-4]
			# wav = f'/home/lynnee/Mandarin_Accent_Conversion/{wav[3:]}'
			wav_scp_list.append(f'{wav_id}\t{wav}')
	wav_scp_list.sort()

	output_file = f'{output_dir}/wav.scp'
	with open(output_file, 'w') as outfile:
		for item in wav_scp_list:
			outfile.writelines(f'{item}\n')


def get_utt2accent(batch_file, output_dir):
	utt2accent_list = []
	with open(batch_file, 'r') as infile:
		for line in infile:
			wav, accent = line.split()
			utt_id = wav.split('/')[-1][:-4]
			utt2accent_list.append(f'{utt_id}\t{accent}')
	utt2accent_list.sort()

	output_file = f'{output_dir}/utt2spk'
	with open(output_file, 'w') as outfile:
		for item in utt2accent_list:
			outfile.writelines(f'{item}\n')


def get_utt2spk(batch_file, output_dir):
	utt2spk_list = []
	with open(batch_file, 'r') as infile:
		for line in infile:
			wav, _ = line.split()
			utt_id = wav.split('/')[-1][:-4]
			spk_id = wav.split('/')[4]
			utt2spk_list.append(f'{utt_id}\t{spk_id}')
	utt2spk_list.sort()

	output_file = f'{output_dir}/utt2spk'
	with open(output_file, 'w') as outfile:
		for item in utt2spk_list:
			outfile.writelines(f'{item}\n')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--partitions_dir', required=True)
	parser.add_argument('--data_dir', required=True)
	parser.add_argument('--task', required=True, choices=['spk', 'acc'])
	args = parser.parse_args()

	DATA_ROOT = args.data_dir

	if args.task == 'spk':
		TRAIN_ROOT = f'{DATA_ROOT}/train'
		TEST_ROOT = f'{DATA_ROOT}/test'
		DEV_ROOT = f'{DATA_ROOT}/dev'
	elif args.task == 'acc':
		TRAIN_ROOT = f'{DATA_ROOT}/train_acc'
		TEST_ROOT = f'{DATA_ROOT}/test_acc'
		DEV_ROOT = f'{DATA_ROOT}/dev_acc'

	PARTITION_ROOT = args.partitions_dir

	if not os.path.exists(DATA_ROOT):
		os.mkdir(DATA_ROOT)
	if not os.path.exists(TRAIN_ROOT):
		os.mkdir(TRAIN_ROOT)
	if not os.path.exists(TEST_ROOT):
		os.mkdir(TEST_ROOT)
	if not os.path.exists(DEV_ROOT):
		os.mkdir(DEV_ROOT)

	# train set
	get_wav_scp(f'{PARTITION_ROOT}/class_train.txt', TRAIN_ROOT)
	if args.task == 'spk':
		get_utt2spk(f'{PARTITION_ROOT}/class_train.txt', TRAIN_ROOT)
	elif args.task == 'acc':
		get_utt2accent(f'{PARTITION_ROOT}/class_train.txt', TRAIN_ROOT)

	# test set
	get_wav_scp(f'{PARTITION_ROOT}/class_test.txt', TEST_ROOT)
	if args.task == 'spk':
		get_utt2spk(f'{PARTITION_ROOT}/class_test.txt', TEST_ROOT)
	elif args.task == 'acc':
		get_utt2accent(f'{PARTITION_ROOT}/class_test.txt', TEST_ROOT)

	# dev set
	get_wav_scp(f'{PARTITION_ROOT}/class_dev.txt', DEV_ROOT)
	if args.task == 'spk':
		get_utt2spk(f'{PARTITION_ROOT}/class_dev.txt', DEV_ROOT)
	elif args.task == 'acc':
		get_utt2accent(f'{PARTITION_ROOT}/class_dev.txt', DEV_ROOT)
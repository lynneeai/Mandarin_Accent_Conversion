import os
import random
from collections import defaultdict

ROOT_DIR = '../aishell_2/data'
SPK_INFO_FILE = f'{ROOT_DIR}/spk_info.txt'
WAV_SCP_FILE = f'{ROOT_DIR}/wav.scp'
SELECT_FROM_NORTH_SEED = 233
SPLIT_CLASS_ENC_SEED = 7
CLASS_TRAIN_TEST_SPLIT_SEED = 77
ENC_TRAIN_TEST_SPLIT_SEED = 777

# output file names
CLASS_TRAIN_FILE = f'{ROOT_DIR}/class_train.txt'
CLASS_TEST_FILE = f'{ROOT_DIR}/class_test.txt'
CLASS_DEV_FILE = f'{ROOT_DIR}/class_dev.txt'
ENC_TRAIN_FILE = f'{ROOT_DIR}/enc_train.txt'
ENC_TEST_FILE = f'{ROOT_DIR}/enc_test.txt'
ENC_DEV_FILE = f'{ROOT_DIR}/enc_dev.txt'

'''--------------Helper Functions--------------'''
'''
Select a random set of k items using a seed.
This is to make sure that we will get the same train/test sets
when running this script multiple times.
'''
def random_k_items(l, k, seed):
	list_length = len(l)
	if k > list_length:
		raise Exception('k should be smaller than list length')
	idx = set()
	random.seed(seed)
	while len(idx) < k:
		idx.add(random.randint(0, list_length - 1))
	return [l[i] for i in idx]

'''
Given a list, find the complement list
'''
def complement_list(l_full, l_given):
	l_given_set = set(l_given)
	return [i for i in l_full if i not in l_given_set]

'''
Write north list and south list together into output file
'''
def write_north_south(output_file, north_list, south_list):
	with open(output_file, 'w') as outfile:
		for item in north_list:
			outfile.writelines(f'{item}\tNorth\n')
		for item in south_list:
			outfile.writelines(f'{item}\tSouth\n')

'''--------------Processing Scripts--------------'''
'''build {speaker:province} dictionary'''
spk_prov_dict = defaultdict(str)
with open(SPK_INFO_FILE, 'r') as infile:
	for line in infile:
		spk, _, _, prov = line.split()
		spk_prov_dict[spk] = prov

'''get North and South lists'''
wav_list_dict = defaultdict(list)
with open(WAV_SCP_FILE, 'r') as infile:
	for line in infile:
		_, wav_file = line.split()
		_, spk, _ = wav_file.split('/')
		prov = spk_prov_dict[spk]
		if prov != 'Other_Areas':
			wav_list_dict[prov].append(f'{ROOT_DIR}/{wav_file}')

'''randomly select from North list to balance data'''
north_len, south_len = len(wav_list_dict['North']), len(wav_list_dict['South'])
print(f'-------randomly select from North list to balance data-------')
print(f'North list #files: {north_len}')
print(f'South list #files: {south_len}')
print(f'Selecting {south_len} files from North list...\n')
north_list = random_k_items(wav_list_dict['North'], south_len, SELECT_FROM_NORTH_SEED)
south_list = wav_list_dict['South']
assert(len(north_list) == len(south_list))

'''split into 4/6 for classifier and autoencoder'''
total_len = len(north_list)
classifier_len = int(total_len * 0.4)
autoencoder_len = total_len - classifier_len
print(f'-------split into 4-6 for classifier and autoencoder-------')
print(f'Total north files: {total_len}')
print(f'Classifier #files: {classifier_len}')
print(f'Autoencoder #files: {autoencoder_len}')
print(f'Same for south list')
print(f'Splitting...\n')
# north list
class_north_list = random_k_items(north_list, classifier_len, SPLIT_CLASS_ENC_SEED)
enc_north_list = complement_list(north_list, class_north_list)
assert(len(class_north_list) == classifier_len)
assert(len(enc_north_list) == autoencoder_len)
# south list
class_south_list = random_k_items(south_list, classifier_len, SPLIT_CLASS_ENC_SEED)
enc_south_list = complement_list(south_list, class_south_list)
assert(len(class_south_list) == classifier_len)
assert(len(enc_south_list) == autoencoder_len)

'''split classifier set into 7/2/1'''
classifier_train_len = int(classifier_len * 0.7)
classifier_test_len = int(classifier_len * 0.2)
classifier_dev_len = classifier_len - classifier_train_len - classifier_test_len
print(f'-------split classifier set into 7/2/1-------')
print(f'Total classifier north files: {classifier_len}')
print(f'Classifier north train #files: {classifier_train_len}')
print(f'Classifier north test #files: {classifier_test_len}')
print(f'Classifier north dev #files: {classifier_dev_len}')
print(f'Same for south list')
print(f'Splitting...\n')
# north list
class_north_train = random_k_items(class_north_list, classifier_train_len, CLASS_TRAIN_TEST_SPLIT_SEED)
temp_comp_list = complement_list(class_north_list, class_north_train)
class_north_test = random_k_items(temp_comp_list, classifier_test_len, CLASS_TRAIN_TEST_SPLIT_SEED)
class_north_dev = complement_list(temp_comp_list, class_north_test)
assert(len(class_north_train) == classifier_train_len)
assert(len(class_north_test) == classifier_test_len)
assert(len(class_north_dev) == classifier_dev_len)
# south list
class_south_train = random_k_items(class_south_list, classifier_train_len, CLASS_TRAIN_TEST_SPLIT_SEED)
temp_comp_list = complement_list(class_south_list, class_south_train)
class_south_test = random_k_items(temp_comp_list, classifier_test_len, CLASS_TRAIN_TEST_SPLIT_SEED)
class_south_dev = complement_list(temp_comp_list, class_south_test)
assert(len(class_south_train) == classifier_train_len)
assert(len(class_south_test) == classifier_test_len)
assert(len(class_south_dev) == classifier_dev_len)

'''split autoencoder set into 7/2/1'''
autoencoder_train_len = int(autoencoder_len * 0.7)
autoencoder_test_len = int(autoencoder_len * 0.2)
autoencoder_dev_len = autoencoder_len - autoencoder_train_len - autoencoder_test_len
print(f'-------split autoencoder set into 7/2/1-------')
print(f'Total autoencoder north files: {autoencoder_len}')
print(f'Autoencoder north train #files: {autoencoder_train_len}')
print(f'Autoencoder north test #files: {autoencoder_test_len}')
print(f'Autoencoder north dev #files: {autoencoder_dev_len}')
print(f'Same for south list')
print(f'Splitting...\n')
# north list
enc_north_train = random_k_items(enc_north_list, autoencoder_train_len, ENC_TRAIN_TEST_SPLIT_SEED)
temp_comp_list = complement_list(enc_north_list, enc_north_train)
enc_north_test = random_k_items(temp_comp_list, autoencoder_test_len, ENC_TRAIN_TEST_SPLIT_SEED)
enc_north_dev = complement_list(temp_comp_list, enc_north_test)
assert(len(enc_north_train) == autoencoder_train_len)
assert(len(enc_north_test) == autoencoder_test_len)
assert(len(enc_north_dev) == autoencoder_dev_len)
# south list
enc_south_train = random_k_items(enc_south_list, autoencoder_train_len, ENC_TRAIN_TEST_SPLIT_SEED)
temp_comp_list = complement_list(enc_south_list, enc_south_train)
enc_south_test = random_k_items(temp_comp_list, autoencoder_test_len, ENC_TRAIN_TEST_SPLIT_SEED)
enc_south_dev = complement_list(temp_comp_list, enc_south_test)
assert(len(enc_south_train) == autoencoder_train_len)
assert(len(enc_south_test) == autoencoder_test_len)
assert(len(enc_south_dev) == autoencoder_dev_len)

'''write to files'''
print(f'Writing to files...\n')
# classifier files
write_north_south(CLASS_TRAIN_FILE, class_north_train, class_south_train)
write_north_south(CLASS_TEST_FILE, class_north_test, class_south_test)
write_north_south(CLASS_DEV_FILE, class_north_dev, class_south_dev)

# autoencoder files
write_north_south(ENC_TRAIN_FILE, enc_north_train, enc_south_train)
write_north_south(ENC_TEST_FILE, enc_north_test, enc_south_test)
write_north_south(ENC_DEV_FILE, enc_north_dev, enc_south_dev)

print(f'Done!')
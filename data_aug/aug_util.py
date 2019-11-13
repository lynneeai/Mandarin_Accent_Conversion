import os
import random
import nlpaug
import nlpaug.augmenter.audio as naa
import librosa


# === loudness ===
def aug_loudness(data, loudness_factor=(0.95, 1.5)):
	aug = naa.LoudnessAug(loudness_factor=loudness_factor)
	augmented_data = aug.augment(data)
	return augmented_data

# === noise ===
def aug_noise(data, noise_factor=0.03):
	aug = naa.NoiseAug(noise_factor=noise_factor)
	augmented_data = aug.augment(data)
	return augmented_data

# === speed ===
def aug_speed(data, speed_range=(0.9, 1.1)):
	aug = naa.SpeedAug(speed_range=speed_range)
	augmented_data = aug.augment(data)
	return augmented_data

# === pitch ===
def aug_pitch(data, pitch_factor=(-2,3)):
	aug = naa.PitchAug(sampling_rate=sr, pitch_factor=pitch_factor)
	augmented_data = aug.augment(data)
	return augmented_data

# === time shift ===
def aug_timeshift(data, sampling_rate):
	aug = naa.ShiftAug(sampling_rate=sr)
	augmented_data = aug.augment(data)
	return augmented_data

# loudness, noise, and speed
def random_aug(file_path, output_dir, n=3):
	file_name = os.path.basename(file_path).split('.')[0]
	out_files = []
	for i in range(1, n+1):
		noise_factor = 0.005 + random.random()*(0.015-0.005)
		#print(noise_factor)
		data, sr = librosa.load(file_path) # data, sampling rate
		print("sampling rate {}".format(sr))
		aug_data = aug_speed(data)
		aug_data = aug_loudness(aug_data)
		aug_data = aug_noise(aug_data, noise_factor=noise_factor)
		out_file = output_dir+file_name+'_aug'+str(i)+'.wav'
		librosa.output.write_wav(out_file, aug_data, sr)
		out_files.append(out_file)
	#print("{} augmentation completed".format(file_name))
	return out_files

if __name__ == "__main__":
	file_path = './input_wav/1.wav'
	output_dir = './aug_wav/'
	random_aug(file_path, output_dir)
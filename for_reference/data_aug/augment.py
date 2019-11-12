import sys
import aug_util

def augment_file_from_list(file_list_path, output_dir, output_file):
    file_list = []
    count = 0
    with open(file_list_path, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for line in infile:
                count += 1
                wav, label = line.strip().split()
                aug_wavs = aug_util.random_aug(wav, output_dir)
                for aug_wav in aug_wavs:
                    outfile.write(f'{aug_wav}\t{label}\n')
                print("file # {} augmented, {}".format(count, wav))
    print("Augmentation complete! {} files augmented".format(count))
            

if __name__ == "__main__":
    if len(sys.argv[1:]) < 3:
        print("Usage: python3 augment.py [wave_file_list] [output_dir] [output_file]")
        exit()
    file_path = sys.argv[1]
    output_dir = sys.argv[2]
    output_file = sys.argv[3]
    augment_file_from_list(file_path, output_dir, output_file)
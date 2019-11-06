import sys
import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms

def trim_silence(audio):
    start_trim = leading_silence(audio)
    end_trim = leading_silence(audio.reverse())
    duration = len(audio)    
    trimmed_audio = audio[start_trim:duration-end_trim]
    return trimmed_audio

def pad_chuck(audio, desired_length_ms):
    ori_ms = len(audio)
    pad_ms = desired_length_ms - ori_ms  # milliseconds of silence needed
    silence = AudioSegment.silent(duration=pad_ms)
    padded = audio + silence  # Adding silence after the audio
    return padded

def chunck_wav(file_path, chunk_length_ms=1000):
    raw_audio = AudioSegment.from_file(file_path, "wav") 
    audio = trim_silence(raw_audio)
    chunks = make_chunks(audio, chunk_length_ms)
    chunks[-1] = pad_chuck(chunks[-1], chunk_length_ms)
    assert(len(chunks[-1]) == chunk_length_ms)
    return chunks

def chunck_wav_files(file_list_path, output_dir, output_file, chunk_length_ms=3000): # pydub calculates in millisec
    with open(file_list_path, 'r') as infile:
        with open(output_file, 'w') as outfile:
            count = 0
            for line in infile:
                count += 1
                wav, label = line.strip().split()
                file_name = os.path.basename(wav).split('.')[0]
                chunks = chunck_wav(wav, chunk_length_ms)
                for i, chunk in enumerate(chunks): # Export chunks as wav files
                    chunk_name = "{}{}_chunk{}.wav".format(output_dir, file_name ,i)
                    chunk.export(chunk_name, format="wav") 
                    outfile.write(f'{chunk_name}\t{label}\n')
                print("file # {} chuncked into {} pieces".format(count, len(chunks)))
    print("Chuncking complete! {} files chuncked".format(count))

if __name__ == "__main__":
    if len(sys.argv[1:]) < 3:
        print("Usage: python3 chunck_wav.py [wave_file_list] [output_dir] [output_file]")
        exit()
    file_path = sys.argv[1]
    output_dir = sys.argv[2]
    output_file = sys.argv[3]
    chunck_wav_files(file_path, output_dir, output_file, chunk_length_ms=1000)

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kaldiio import ReadHelper
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ACCENTS = ['chuan', 'dongbei', 'guan', 'wu', 'yue']
ACCENT_IDX = {'chuan':0, 'dongbei':1, 'guan':2, 'wu':3, 'yue':4}

SCP_FILE = 'exp/trans_nnet/xvectors_test_magicdata/xvector.scp'
UTT2ACC_FILE = 'data/test_magicdata/utt2spk'
WAV_SCP_FILE = 'data/test_magicdata/wav.scp'

preds = []
preds_dict = {}
with ReadHelper(f'scp:{SCP_FILE}') as reader:
	for utt, logsoftmax in reader:
		idx = np.argmax(logsoftmax)
		preds.append((utt, idx))
		preds_dict[utt] = ACCENTS[idx]
preds.sort(key=lambda x: x[0])
preds = [idx for utt, idx in preds]

targets = []
targets_dict = {}
with open(UTT2ACC_FILE, 'r') as infile:
	for line in infile:
		utt, acc = line.split()
		targets.append((utt, ACCENT_IDX[acc]))
		targets_dict[utt] = acc
targets.sort(key=lambda x: x[0])
targets = [idx for utt, idx in targets]

assert(len(preds) == len(targets))

targets_label = [ACCENTS[i] for i in targets]
preds_label = [ACCENTS[i] for i in preds]

accuracy = accuracy_score(targets, preds)
f1_macro = f1_score(targets, preds, average='macro')
f1_micro = f1_score(targets, preds, average='micro')
confusion_mat = confusion_matrix(targets_label, preds_label, labels=ACCENTS)

print(f'Magicdata Test Accuracy: {accuracy}')
print(f'Magicdata Test macro F1 score: {f1_macro}')
print(f'Magicdata Test micro F1 score: {f1_micro}')
print(f'Magicdata Confusion Matrix: \n{confusion_mat}')

df_cm = pd.DataFrame(confusion_mat, index = ACCENTS, columns = ACCENTS)
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
res = sns.heatmap(df_cm, annot=True, cmap=cmap)
plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5], ACCENTS, va='center')
res.set_ylim(5, 0)
res.invert_yaxis()
plt.title('Magicdata Test Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=400, bbox_inches='tight')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--utt', type=str, required=False)
	args = parser.parse_args()

	if args.utt != None:
		utt_path_dict = {}
		with open(WAV_SCP_FILE, 'r') as infile:
			for line in infile:
				utt, path = line.split()
				utt_path_dict[utt] = path
		print(f'-------Utterance {args.utt} results-------')
		print(f'True accent: {targets_dict[args.utt]}')
		print(f'Predicted accent: {preds_dict[args.utt]}')
		print(f'Original audio path: {utt_path_dict[args.utt]}')
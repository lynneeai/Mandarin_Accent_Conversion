# -*- coding: utf-8 -*-
"""Classifier_Attempts

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xo1fCXFdhJLQpxYJ71VNltzkr3_KYEC6

# Author: Lin Ai (la2734)
# This notebook is an attempt of building RNN-based, TDNN-based, and CNN-based accent classifiers

# Google Cloud Storage (GCS)
"""

project_id = 'speechrec-255319'
bucket_name = 'aishell_2'

from google.colab import auth
auth.authenticate_user()

!gcloud config set project {project_id}

"""# Preparation

## Import Packages
"""

import os
import pickle
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support

"""## Download Data"""

if not os.path.exists('/tmp/data'):
    os.mkdir('/tmp/data')
!gsutil -m cp -r gs://aishell_2/sp_data/class_train /tmp/data/
!gsutil -m cp -r gs://aishell_2/sp_data/class_test /tmp/data/

"""# RNN Model"""

class LSTMClassifier(nn.Module):

    def __init__(self, sp_dim, label_size, hidden_dim, lstm_num_layers, batch_size, use_gpu=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.batch_size = batch_size
        self.feature_dim = sp_dim
        self.label_size = label_size
        self.use_gpu = use_gpu

        self.lstm = nn.LSTM(input_size=sp_dim, hidden_size=hidden_dim, 
                            num_layers=lstm_num_layers, bidirectional=False, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.BCEWithLogitsLoss()

    def init_hidden(self):
        if self.use_gpu:
            h0 = torch.autograd.Variable(torch.zeros(self.lstm_num_layers, self.batch_size, self.hidden_dim).cuda())
            c0 = torch.autograd.Variable(torch.zeros(self.lstm_num_layers, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = torch.autograd.Variable(torch.zeros(self.lstm_num_layers, self.batch_size, self.hidden_dim))
            c0 = torch.autograd.Variable(torch.zeros(self.lstm_num_layers, self.batch_size, self.hidden_dim))
        return (h0.detach(), c0.detach())

    def forward(self, sp):
        lstm_out, (hn, cn) = self.lstm(sp, self.hidden)
        y  = self.softmax(self.hidden2label(lstm_out[:, -1, :]))
        return y

    def loss(self, predict, label):
        return self.criterion(predict, label)

"""# TDNN Model"""

'''refer to https://github.com/SiddGururani/Pytorch-TDNN'''
"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""

class TDNN(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context=False, use_gpu=False):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TDNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context,full_context)
        self.register_buffer('context',torch.LongTensor(context))
        self.full_context = full_context
        stdv = 1./math.sqrt(input_dim)
        self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0,stdv))
        self.use_gpu = use_gpu

    def forward(self,x):
        """
        x is one batch of data
        x.size(): [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features
        output size: [batch_size, output_dim, len(valid_steps)]
        """
        # Check if parameters are cuda type and change context
        if self.use_gpu:
            self.context = self.context.cuda()
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        return F.relu(conv_out)

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context. Cannot directly use convolution because in case of only particular frames of context,
        one needs to select only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
        """
        input_size = x.size()
        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [batch_size, input_sequence_length, input_dim] = input_size
        x = x.transpose(1,2).contiguous()

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        xs = Variable(self.bias.data.new(batch_size, kernel.size()[0], len(valid_steps)))
        if self.use_gpu:
            xs = xs.cuda()

        # Perform the convolution with relevant input frames
        for c, i in enumerate(valid_steps):
            if self.use_gpu:
                features = torch.index_select(x, 2, context+i).cuda()
                kernel = kernel.cuda()
                bias = bias.cuda()
                xs[:,:,c] = F.conv1d(features, kernel, bias = bias)[:,:,0].cuda()
            else:
                features = torch.index_select(x, 2, context+i)
                xs[:,:,c] = F.conv1d(features, kernel, bias = bias)[:,:,0]
        return xs

    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0],context[-1]+1)
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -1*context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)

class TDNNClassifier(nn.Module):

    def __init__(self, sp_dim, label_size, context_layers, output_sizes, batch_size, use_gpu=False):
        super(TDNNClassifier, self).__init__()
        self.input_dim = sp_dim
        self.output_dim = label_size
        self.context_layers = context_layers
        self.output_sizes = output_sizes
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        self.tdnn1 = TDNN(context_layers[0], self.input_dim, output_sizes[0], use_gpu=use_gpu)
        self.tdnn2 = TDNN(context_layers[1], output_sizes[0], output_sizes[1], use_gpu=use_gpu)
        self.tdnn3 = TDNN(context_layers[2], output_sizes[1], output_sizes[2], use_gpu=use_gpu)
        self.tdnn4 = TDNN(context_layers[3], output_sizes[2], output_sizes[3], use_gpu=use_gpu)
        self.tdnn5 = TDNN(context_layers[4], output_sizes[3], output_sizes[4], use_gpu=use_gpu)
        # TODO: stats pooling layer
        self.tdnn6 = TDNN(context_layers[5], output_sizes[4], output_sizes[5])
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sp):
        frame1 = self.tdnn1(sp).view(self.batch_size, -1, self.output_sizes[0])
        frame2 = self.tdnn2(frame1).view(self.batch_size, -1, self.output_sizes[1])
        frame3 = self.tdnn3(frame2).view(self.batch_size, -1, self.output_sizes[2])
        frame4 = self.tdnn4(frame3).view(self.batch_size, -1, self.output_sizes[3])
        frame5 = self.tdnn5(frame4).view(self.batch_size, -1, self.output_sizes[4])
        frame6 = self.tdnn6(frame5).view(self.batch_size, -1, self.output_sizes[5])
        output = frame6.view(self.batch_size, -1)
        
        fc = nn.Linear(output.size()[1], 2)
        if self.use_gpu:
            fc = fc.cuda()

        output = fc(output)
        y = self.softmax(output)

        return y

    def loss(self, predict, label):
        return self.criterion(predict, label)

"""# CNN Model"""

class CNNClassifier(nn.Module):

    def __init__(self, sp_dim, label_size, batch_size, use_gpu=False):
        super(CNNClassifier, self).__init__()
        self.input_dim = sp_dim
        self.output_dim = label_size
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=82560, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=label_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sp):
        # (N,C,H,W)
        x = sp.view(self.batch_size, 1, self.input_dim, -1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        output = self.softmax(self.fc2(x))
        return output
    
    def loss(self, predict, label):
        return self.criterion(predict, label)

"""# Train Model

## DataLoader
"""

class DataLoader:

    def __init__(self, dataset_name, batch_size, total_chunks, chunks_to_load, min_len, fixed_len=None):
        self.data_root = '/tmp/data'
        self.dataset_name = dataset_name
        self.partition_dir = f'{self.data_root}/{self.dataset_name}'

        self.batch_size = batch_size
        self.total_chunks = total_chunks
        self.chunks_to_load = chunks_to_load
        self.min_len = min_len
        self.fixed_len = fixed_len

        self.batch_ptr = 0
        self.chunk_ptr = 0

        self.epoch_done = False

        self.load_data()

    def load_data(self):
        print(f'start loading {self.dataset_name} from chunk {self.chunk_ptr}')
        utt_id_list = []
        f0_list = []
        sp_list = []
        ap_list = []
        label_list = []
        for i in range(self.chunk_ptr, self.chunk_ptr + self.chunks_to_load):
            # load f0, sp, ap
            pw_list = np.load(f'{self.partition_dir}/pw_{i}.npz', allow_pickle=True)
            f0_list.extend(pw_list['f0'])
            sp_list.extend(pw_list['sp'])
            ap_list.extend(pw_list['ap'])

            # load utt_id, label
            with open(f'{self.partition_dir}/utt_label_{i}.pkl', 'rb') as utt_label_file:
                utt_ids, labels = pickle.load(utt_label_file)
                utt_id_list.extend(utt_ids)
                label_list.extend(labels)

        # filter out samples that are too short
        idx_to_remove = set([i for i, f0 in enumerate(f0_list) if f0.shape[0] < self.min_len])
        utt_id_list = [item for i, item in enumerate(utt_id_list) if i not in idx_to_remove]
        f0_list = [item for i, item in enumerate(f0_list) if i not in idx_to_remove]
        sp_list = [item for i, item in enumerate(sp_list) if i not in idx_to_remove]
        ap_list = [item for i, item in enumerate(ap_list) if i not in idx_to_remove]
        label_list = [item for i, item in enumerate(label_list) if i not in idx_to_remove]   

        self.utt_id_list = utt_id_list
        self.f0_list = f0_list
        self.sp_list = sp_list
        self.ap_list = ap_list
        self.label_list = label_list

        self.loaded_wavs = len(self.utt_id_list)
        self.update_chunk_ptr()
    
    def next_batch(self):
        total = self.loaded_wavs
        size = self.batch_size
        start = self.batch_ptr
        end = min(start + size, total)
        
        utt_ids = self.utt_id_list[start:end]
        f0s = self.f0_list[start:end]
        sps = self.sp_list[start:end]
        aps = self.ap_list[start:end]
        labels = self.label_list[start:end]
        self.batch_ptr = end

        if start + size >= total:
            self.load_data()
            self.batch_ptr = 0
            if start + size > total:
                shortage = start + size - end
                new_start = self.batch_ptr
                new_end = new_start + shortage

                utt_ids.extend(self.utt_id_list[new_start:new_end])
                f0s.extend(self.f0_list[new_start:new_end])
                sps.extend(self.sp_list[new_start:new_end])
                aps.extend(self.ap_list[new_start:new_end])
                labels.extend(self.label_list[new_start:new_end])
                self.batch_ptr = new_end	

            if self.chunk_ptr == self.chunks_to_load:
                self.epoch_done = True

        f0s = self.validate_batch(f0s)
        sps = self.validate_batch(sps)
        aps = self.validate_batch(aps)
        labels = self.get_label_array_onehot(labels)

        return utt_ids, f0s, sps, aps, labels
    
    def update_chunk_ptr(self):
        self.chunk_ptr += self.chunks_to_load
        if self.chunk_ptr == self.total_chunks:
            self.chunk_ptr = 0
        if self.chunk_ptr + self.chunks_to_load > self.total_chunks:
            self.chunk_ptr = self.total_chunks - self.chunks_to_load
    
    def is_epoch_done(self):
        return self.epoch_done
    
    def next_epoch(self):
        self.batch_ptr = 0
        self.epoch_done = False

    def validate_batch(self, batch_data): 
        batch_data = [i.reshape((i.shape[0], -1)) for i in batch_data]
        feature_size = batch_data[0].shape[1]
        
        if self.fixed_len != None:
            desired_len = self.fixed_len
        else:
            data_lens = [i.shape[0] for i in batch_data]
            desired_len = int(np.mean(data_lens))
        
        # truncate or pad sample
        new_data = []
        for sample in batch_data:
            sample = sample[:desired_len]
            if sample.shape[0] < desired_len:
                pad = np.zeros((desired_len, feature_size))
                pad[:sample.shape[0], :feature_size] = sample
                sample = pad
            new_data.append(sample)
        return np.array(new_data).reshape((self.batch_size, desired_len, feature_size))

    def get_label_array_onehot(self, labels):
        label_idx = {'North':0, 'South':1}
        label_array = np.zeros((self.batch_size, 2))
        for i, l in enumerate(labels):
            label_array[i, label_idx[l]] = 1.0
        return label_array
    
    def get_label_array_category(self, labels):
        label_idx = {'North':0, 'South':1}
        label_array = np.zeros(self.batch_size)
        for i, l in enumerate(labels):
            label_array[i] = label_idx[l]
        return label_array

"""## Train"""

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
    
def calculate_precision_recall(preds, labels):
    _, predicts= torch.max(preds.data, 1)
    _, targets = torch.max(labels, 1)
    predicts = predicts.cpu().numpy()
    targets = targets.cpu().numpy()
    precision, recall, fscore, _ = precision_recall_fscore_support(targets, predicts, average='micro')
    return precision, recall

def save_results(root, model, config, results):
    if not os.path.exists(root):
        os.mkdir(root)

    model_filepath = f'{root}/statedict.pth'
    torch.save(model.state_dict(), model_filepath)


    config_filepath = f'{root}/config.pkl'
    param = {}
    param['model_type'] = config.model_type
    param['epochs'] = config.epochs
    param['batch_size'] = config.batch_size
    param['total_chunks'] = config.total_chunks
    param['chunks_to_load'] = config.chunks_to_load
    param['learning_rate'] = config.learning_rate
    param['lstm_num_layers'] = config.lstm_num_layers
    param['hidden_dim'] = config.hidden_dim

    result = {}
    result['train_loss'] = results['train_loss']
    result['train_precision'] = results['train_precision']
    result['train_recall'] = results['train_recall']
    result['test_loss'] = results['test_loss']
    result['test_precision'] = results['test_precision']
    result['test_recall'] = results['test_recall']
    result['param'] = param

    with open(config_filepath, 'wb') as outfile:
        pickle.dump(result, outfile)

def plot_results(root, results):
    epoch_num = len(results['train_loss'])
    epochs = [i + 1 for i in range(epoch_num)] + [i + 1 for i in range(epoch_num)]
    dsets = ['train' for i in range(epoch_num)] + ['test' for i in range(epoch_num)]

    # loss plot
    loss_values = [i.data.cpu().tolist() for i in results['train_loss']] + [i.data.cpu().tolist() for i in results['test_loss']]
    print(epochs)
    print(dsets)
    print(loss_values)
    loss_df = pd.DataFrame({'epoch':epochs, 'dset':dsets, 'loss':loss_values})
    ax = sns.lineplot(x='epoch', y='loss', hue='dset', data=loss_df)
    plt.savefig(f'{root}/loss.png')
    plt.close()

    # precision plot
    precision_values = results['train_precision'] + results['test_precision']
    precision_df = pd.DataFrame({'epoch':epochs, 'dset':dsets, 'precision':precision_values})
    ax = sns.lineplot(x='epoch', y='precision', hue='dset', data=precision_df)
    plt.savefig(f'{root}/precision.png')
    plt.close()

    # recall plot
    recall_values = results['train_recall'] + results['test_recall']
    recall_df = pd.DataFrame({'epoch':epochs, 'dset':dsets, 'recall':recall_values})
    ax = sns.lineplot(x='epoch', y='recall', hue='dset', data=recall_df)
    plt.savefig(f'{root}/recall.png')
    plt.close()

class Config:
    '''Training params'''
    epochs = 10
    batch_size = 64
    total_chunks = 21
    chunks_to_load = 3
    learning_rate = 0.01
    min_len = 250
    fixed_len = 350
    use_gpu = True
    model_type = 'TDNN'
    sp_dim = 129
    label_size = 2

    '''RNN Model params'''
    lstm_num_layers = 2
    hidden_dim = 64

    '''TDNN Model params'''
    context_layers = [[-2, -1, 0, 1, 2],
                      [-2, 0, 2],
                      [-3, 0, 3],
                      [0],
                      [0],
                      [0]]
    output_sizes = [512, 512, 512, 512, 1500, 512]

    '''CNN Model params'''

'''Training process'''

config = Config()

# model = LSTMClassifier(hidden_dim=config.hidden_dim, lstm_num_layers=config.lstm_num_layers,
#                        sp_dim=config.sp_dim, label_size=config.label_size, 
#                        batch_size=config.batch_size, use_gpu=config.use_gpu)
model = TDNNClassifier(sp_dim=config.sp_dim, label_size=config.label_size, 
                       context_layers=config.context_layers, output_sizes=config.output_sizes, 
                       batch_size=config.batch_size, use_gpu=config.use_gpu)
# model = CNNClassifier(sp_dim=config.sp_dim, label_size=config.label_size, 
#                      batch_size=config.batch_size, use_gpu=config.use_gpu)
if config.use_gpu:
    model = model.cuda()

train_dset = DataLoader('class_train', batch_size=config.batch_size, 
                        total_chunks=config.total_chunks, chunks_to_load=config.chunks_to_load, 
                        min_len=config.min_len, fixed_len=config.fixed_len)
test_dset = DataLoader('class_test', batch_size=config.batch_size, 
                       total_chunks=config.total_chunks, chunks_to_load=config.chunks_to_load,
                       min_len=config.min_len, fixed_len=config.fixed_len)
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

train_loss = []
train_precision = []
train_recall = []
test_loss = []
test_precision = []
test_recall = []

for epoch in range(config.epochs):

    '''Training...'''
    print(f'[Training epoch {epoch}/{config.epochs - 1}...]')
    # optimizer = adjust_learning_rate(optimizer, epoch)
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_batches = 0.0

    while not train_dset.is_epoch_done():
        utt_ids, f0s, sps, aps, labels = train_dset.next_batch()
        sps = torch.Tensor(sps).cuda() if config.use_gpu else torch.Tensor(sps)
        labels = torch.Tensor(labels).cuda() if config.use_gpu else torch.Tensor(labels)

        optimizer.zero_grad()
        preds = model(sps)

        loss = model.loss(preds, labels)
        loss.backward()
        optimizer.step()

        precision, recall = calculate_precision_recall(preds, labels)
        total_loss += loss.data
        total_precision += precision
        total_recall += recall
        total_batches += 1
    
    train_loss.append(total_loss / total_batches)
    train_precision.append(total_precision / total_batches)
    train_recall.append(total_recall / total_batches)

    '''Testing...'''
    print(f'[Testing epoch {epoch}/{config.epochs - 1}...]')
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_batches = 0.0
    
    while not test_dset.is_epoch_done():
        utt_ids, f0s, sps, aps, labels = test_dset.next_batch()
        sps = torch.Tensor(sps).cuda() if config.use_gpu else torch.Tensor(sps)
        labels = torch.Tensor(labels).cuda() if config.use_gpu else torch.Tensor(labels)

        preds = model(sps)

        loss = model.loss(preds, labels)

        precision, recall = calculate_precision_recall(preds, labels)
        total_loss += loss.data
        total_precision += precision
        total_recall += recall
        total_batches += 1
    
    test_loss.append(total_loss / total_batches)
    test_precision.append(total_precision / total_batches)
    test_recall.append(total_recall / total_batches)

    print(f'-------[Epoch: {epoch}/{config.epochs-1}]-------')
    print(f'Training Loss: {train_loss[epoch]}, Training Precision: {train_precision[epoch]}, Training Recall: {train_recall[epoch]}')
    print(f'Tesing Loss: {test_loss[epoch]}, Testing Precision: {test_precision[epoch]}, Testing Recall: {test_recall[epoch]}')

    train_dset.next_epoch()
    test_dset.next_epoch()

results = {'train_loss':train_loss, 'train_precision':train_precision, 'train_recall':train_recall,
           'test_loss':test_loss, 'test_precision':test_precision, 'test_recall':test_recall}
if not os.path.exists('/tmp/models'):
    os.mkdir('/tmp/models')
checkpoint_time = datetime.now().strftime("%d-%h-%m-%s")
save_root = f'/tmp/models/{config.model_type}_{checkpoint_time}'

save_results(save_root, model, config, results)
plot_results(save_root, results)

'''upload to GCS'''
cmd = f'gsutil -m cp -r {save_root} gs://aishell_2/lin_models/'
os.system(cmd)


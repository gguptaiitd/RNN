from __future__ import print_function, division
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import torch.utils.data as data_utils
from Bio import SeqIO
import random
import operator
series_length = 10
refSeries = 4000
mutSeries = 10
mutSeries_length = int(series_length*0.4)
unmatched_seq_length = mutSeries_length + 1
input_series_length = refSeries*(mutSeries_length+unmatched_seq_length)

nucleotide2num = dict(zip("ACGT",range(4)))
class mdict(dict):
    def __setitem__(self, key, value):
        """add the given value to the list of values for this key"""
        self.setdefault(key, []).append(value)

def generateHotVectorEncoding(data):
    label = list(map(lambda x: nucleotide2num[x], data))
    encoded = to_categorical(label, 4)
    return encoded

# Creates the dataset
def generateData():
    file = open("/media/hp/BackUp/ecoli_exp/sequence.fasta", "r")
    for r in SeqIO.parse(file, "fasta"):
    #for r in FastaReader(file):
        # if r.name != ctg_name:
        #    continue
        ref_seq = r.seq

    ref_series = {}
    mut_series = mdict()
    loc = random.sample(list(range(0, len(ref_seq)-series_length)), refSeries)
    for loc in loc:
        l_seq = series_length
        seq = list(ref_seq[loc:loc + l_seq])
        if (str(seq)).find('N') !=- 1:
            print ("N Detected ")
            continue
        h_r_seq = generateHotVectorEncoding(seq)
        ref_series[loc] = h_r_seq
        s_seed = [random.randint(0, l_seq-mutSeries_length) for _ in range(mutSeries)]
        s_rlen = [random.randint(1, mutSeries_length) for _ in range(len(s_seed))]
        for start, l in zip(s_seed, s_rlen):
            s_seq = seq[:]
            subs = []
            [subs.append(random.choice('ACGT')) for _ in range(l)]
            s_seq[start:start + l] = subs
            h_m_seq = generateHotVectorEncoding(s_seq)
            mut_series[loc] = h_m_seq
    keys = list(ref_series.keys())
    random.shuffle(keys)
    input_X = []
    input_Y = []
    for loc in keys:
        x = []
        y =[]
        for i in range(mutSeries):
            x.append([ref_series[loc], mut_series[loc][i]])
            y.append([1, 0])
        # y= np.ones((mutSeries_length))
        for i in range(unmatched_seq_length):
            loc_key = keys[:]
            loc_key.remove(loc)
            w_loc = random.choice(loc_key)
            x.append([ref_series[loc], ref_series[w_loc]])
            y.append([0, 1])
        # y = np.append(np.zeros((unmatched_seq_length)), y)

        r = random.random()
        random.shuffle(x, lambda: r)
        random.shuffle(y, lambda: r)
        input_X.append(x)
        input_Y.append(y)
    return input_X, input_Y


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nn.Sigmoid)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, input):
        input = input.view(-1, batch_size, 8)
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        dim = input.shape

        # Set initial states
        # if self.bidirectional:
        #     h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        # else:
        # combined = torch.cat((input, hidden), 2)
        # Forward propagate RNN
        out, hidden = self.rnn(input, h0)
        output = self.fc(out)
        output = output.view(batch_size, -1, 2)
        # output = []
        # # Decode hidden state of last time step
        # dim = out.shape
        # for i in range(dim[0]):
        #     output.append(self.fc(out[i]))
        output = torch.sum(torch.stack(output), dim=1)
        return softmax(output)

num_epochs = 100
total_series_length = input_series_length
truncated_backprop_length = 10
state_size = 4
num_classes = 2 #matched/ not matched
batch_size = 10
rep_size = 8
num_batches = total_series_length//batch_size//truncated_backprop_length



# model = nn.Sequential()
# model.add_module('DNN ', RNN(8, 4,1))
# for current_input in inputs_series:
#     current_input = tf.reshape(current_input, [batch_size, 8])
#     input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns
#
#     next_state = tf.nn.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
#     states_series.append(next_state)
#     current_state = next_state

# logits_series = [tf.nn.sigmoid(tf.matmul(state, W2) + b2) for state in states_series] #Broadcasted addition
# r_logits = [tf.convert_to_tensor(logits_series, dtype=tf.float32)[:, i, :] for i in range(batch_size)]
# logits = tf.reduce_sum(tf.convert_to_tensor(r_logits), axis=1)
# # p_series = [tf.nn.softmax(logits) for logits in logits_series]
# predictions_series = tf.nn.softmax(logits)
#
# losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_series)
# total_loss = tf.reduce_mean(losses)
#
# train_step = tf.train.RMSPropOptimizer(0.001).minimize(total_loss)

# correct_list = []

def plot(loss_list, correct_list, batchX, batchY):
    # correct =0
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    plt.subplot(2, 3, 2)
    plt.cla()
    plt.plot(correct_list)
    plt.draw()
    plt.pause(0.0001)



plt.ion()
plt.figure()
plt.show()
loss_list = []
correct_list = []
correct_list_test = []
test_loss = 0
correct = 0
rnn = RNN(rep_size, state_size, 1)
optimizer = optim.RMSprop(rnn.parameters(), lr=0.001)
for epoch_idx in range(num_epochs):
    x, y = generateData()
    _current_state = np.zeros((batch_size, state_size))

    print("New data, epoch", epoch_idx)
    softmax = nn.Softmax()
    for batch_idx in range(num_batches):
        start_idx = random.randint(0, len(x)-1)

        # print(start_idx)
        r = random.random()
        random.shuffle(x[start_idx], lambda: r)
        random.shuffle(y[start_idx], lambda: r)
        X = np.asarray(x[start_idx][0:batch_size])
        batchY = y[start_idx][0:batch_size]
        if(X[0][0].shape != X[0][1].shape):
            print("Shape mismatch ", X[0][0].shape, " ", X[0][1].shape)
        batchX = np.asarray([np.concatenate((A[0], A[1]), axis=1) for A in X])
        traindata = data_utils.TensorDataset(
            torch.from_numpy(np.asarray(X, dtype=np.float32)).double(),
            torch.from_numpy(np.asarray(batchY, dtype=np.float32)).double())
        optimizer.zero_grad()
        dataloader = data_utils.DataLoader(traindata, batch_size, shuffle=True)
        input = Variable(torch.from_numpy(np.asarray(batchX, dtype=np.float32)).double()).float()
        target = Variable(torch.from_numpy(np.asarray(batchY, dtype=np.float32)).double()).float()
        # norm = input.norm(p=2, dim=2, keepdim=True)
        # input_normalized = input.div(norm)
        h0 = Variable(torch.zeros(batch_size, state_size))
        output = rnn(input)
        test_loss = F.binary_cross_entropy_with_logits(output, target)
        pred = softmax(output).data.max(1)[1]
        correct = pred.eq(target.data.max(1)[1]).cpu().sum()
        correct_list.append(float(correct*100/batch_size))
        test_loss.backward()
        optimizer.step()
        loss_list.append(test_loss.data[0])

        if batch_idx % 100 == 0:
            print("Step", batch_idx, "Loss", test_loss)
            print("Step", batch_idx, "Correct", correct*100/batch_size)
            # print("Weights RNN FC", rnn.fc.weight.data)
            plot(loss_list, correct_list, batchX, batchY)
    if epoch_idx % 2 == 0:
        # print("Test Weights RNN FC", rnn.fc.weight.data)
        rnn.eval()
        test_loss = 0
        correct = 0
        start_idx = random.randint(0, len(x) - 1)

        # print(start_idx)
        r = random.random()
        random.shuffle(x[start_idx], lambda: r)
        random.shuffle(y[start_idx], lambda: r)
        X = np.asarray(x[start_idx][0:batch_size])
        batchY = y[start_idx][0:batch_size]
        if (X[0][0].shape != X[0][1].shape):
            print("Shape mismatch ", X[0][0].shape, " ", X[0][1].shape)
        input = Variable(torch.from_numpy(np.asarray(batchX, dtype=np.float32)).double()).float()
        target = Variable(torch.from_numpy(np.asarray(batchY, dtype=np.float32)).double()).float()
        # norm = input.norm(p=2, dim=2, keepdim=True)
        # input_normalized = input.div(norm)
        output = rnn(input)
        pred = softmax(output).data.max(1)[1]
        correct = pred.eq(target.data.max(1)[1]).cpu().sum()
        correct_list_test.append(float(correct * 100 / batch_size))
        print("TesT Accuracy ", float(correct * 100 / batch_size))
        print(" Mean TesT Accuracy ", sum(correct_list_test)/len(correct_list_test))
        plt.subplot(2, 3, 3)
        plt.cla()
        plt.plot(correct_list_test)

plt.ioff()
plt.show()
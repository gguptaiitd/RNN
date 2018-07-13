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
series_length = 15
refSeries = 4000
mutSeries = 10
mutSeries_length = int(15*0.4)
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
        for i in range(mutSeries_length):
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

def plot(loss_list, predictions_series, batchX, batchY):
    correct =0
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    plt.subplot(2, 3, 3)
    plt.cla()
    plt.plot(correct_list)
    # plt.plot(_predictions_series)
    # plt.plot(batchY)

    for batch_series_idx in range(batch_size):
        # one_hot_output_series = np.array(predictions_series)[:, batch_series_idx]
        single_output_series = np.array([(1 if out > 0.5 else 0) for out in predictions_series[batch_series_idx]])
        index_output = max(enumerate(single_output_series), key=operator.itemgetter(1))
        index_target = max(enumerate(batchY[batch_series_idx]), key=operator.itemgetter(1))
        if index_output == index_target:
            correct += 1
        plt.subplot(3, 3, 2)
        plt.cla()
        # plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(2)
        # plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx], width=1, color="red")
        plt.bar(left_offset, single_output_series*0.5, width=1, color="green")
    correct_list.append(float(correct*100/batch_size))
    print(float(correct*100/batch_size))
    plt.draw()
    plt.pause(0.0001)


num_epochs = 100
total_series_length = input_series_length
truncated_backprop_length = 15
state_size = 4
num_classes = 2 #matched/ not matched
batch_size = 10
rep_size = 8
num_batches = total_series_length//batch_size//truncated_backprop_length


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, rep_size])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, num_classes])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])




with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x, y = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

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
            # print("batchX shape", batchX.shape)
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
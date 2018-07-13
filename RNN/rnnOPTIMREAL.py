from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import time
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from numpy import array
from Bio import SeqIO
import random
import operator
series_length = 15
refSeries = 4000
mutSeries = 10
mutSeries_length = int(series_length*0.6)
insertSeries_length = int(series_length*0.2)

unmatched_seq_length = mutSeries + 250
input_series_length = refSeries*(mutSeries_length+unmatched_seq_length)
real_unmatched = 5
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
def generateDataMutate():
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

def generateDataInsert():
    file = open("/media/hp/BackUp/ecoli_exp/sequence.fasta", "r")
    for r in SeqIO.parse(file, "fasta"):
        ref_seq = r.seq

    ref_series = {}
    insert_series = mdict()
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
        s_rlen = [random.randint(1, insertSeries_length) for _ in range(len(s_seed))]
        for start, l in zip(s_seed, s_rlen):
            s_seq = seq[:]
            l = len(s_seq)
            insert = []
            [insert.append(random.choice('ACGT')) for _ in range(l)]
            s_seq[start] = insert
            s_seq = s_seq[0:l]
            h_m_seq = generateHotVectorEncoding(s_seq)
            insert_series[loc] = h_m_seq
    keys = list(ref_series.keys())
    random.shuffle(keys)
    input_X = []
    input_Y = []
    for loc in keys:
        x = []
        y =[]
        for i in range(mutSeries_length):
            x.append([ref_series[loc], insert_series[loc][i]])
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

def readRealData():
    import pysam
    real_query = []
    real_ref = {}
    x = []
    y = []
    l = int(series_length/2)-1
    m = series_length - l
    count = 0
    bamfile = pysam.AlignmentFile("/media/hp/BackUp/ecoli_exp/SRR2724094_1/SRR2724094_1.bam", "rb")
    for read in bamfile.fetch():
        if read.reference_start != -1:
            count += 1
            for (qu_pos, ref_pos) in read.aligned_pairs:
                if ref_pos is not None :
                    break
            real_query.append([read.query_sequence, read.blocks, (qu_pos,ref_pos)])
        if count > 1000:
            break
    bamfile.close()
    file = open("/media/hp/BackUp/ecoli_exp/sequence.fasta", "r")
    for r in SeqIO.parse(file, "fasta"):
        ref_seq = r.seq
    file.close()
    for query_list in real_query:
        r_low, r_max = query_list[1][0]
        qu_pos, ref_pos = query_list[2]
        ref = ref_seq[r_low:r_low + series_length]
        if (r_max - r_low) > series_length:
            if qu_pos - l < 0:
                l = 0
                m = series_length
            r_hot = generateHotVectorEncoding(ref)
            q_hot = generateHotVectorEncoding(query_list[0][qu_pos-l:qu_pos+m])
            x.append([r_hot, q_hot])
            y.append([1, 0])
        unmatched_query = random.sample(real_query, real_unmatched)
        if query_list in unmatched_query:
            unmatched_query.remove(query_list)
        for u_q in unmatched_query:
            u_hot = generateHotVectorEncoding(u_q[0][:series_length])
            x.append([r_hot, u_hot])
            y.append([0, 1])
        r = random.random()
        random.shuffle(x, lambda: r)
        random.shuffle(y, lambda: r)
    r = random.random()
    random.shuffle(x, lambda: r)
    random.shuffle(y, lambda: r)
    return x, y


num_epochs = 30
total_series_length = input_series_length
truncated_backprop_length = series_length
state_size = 7
num_classes = 2 #matched/ not matched
batch_size = 10
rep_size = 8
num_batches = total_series_length//batch_size//truncated_backprop_length


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, rep_size])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, num_classes])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(rep_size+state_size, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

W3 = tf.Variable(np.random.rand(batch_size, truncated_backprop_length, num_classes), dtype=tf.float32)
b3 = tf.Variable(np.zeros((batch_size, 1)), dtype=tf.float32)
keep_prob = tf.placeholder(tf.float32)
saver = tf.train.Saver()

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = batchY_placeholder

# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 8])
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # Increasing number of columns
    # keep_prob = tf.placeholder(tf.float32)
    next_state = tf.nn.sigmoid(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    # drop_out = tf.nn.dropout(next_state, keep_prob)
    drop_out = tf.layers.dropout(next_state, 0.2653750429101209)
    states_series.append(drop_out)
    current_state = next_state

logits_series = [tf.nn.sigmoid(tf.matmul(state, W2) + b2) for state in states_series] #Broadcasted addition
r_logits = [tf.convert_to_tensor(logits_series, dtype=tf.float32)[:, i, :] for i in range(batch_size)]
logits = tf.reduce_sum(tf.convert_to_tensor(r_logits), axis=1)
# p_series = [tf.nn.softmax(logits) for logits in logits_series]
predictions_series = tf.nn.softmax(logits)

losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_series)
total_loss = tf.reduce_mean(losses)

train_step = tf.train.RMSPropOptimizer(0.095).minimize(total_loss)

correct_predictions = tf.equal(tf.argmax(predictions_series, 1), tf.cast(tf.argmax(labels_series,1), tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

correct_list = []
accuracy_list = []
test_accuracy_list = []
pred_list = []
target_list = []
def plot(loss_list, predictions_series, batchX, batchY):
    correct = 0
    plt.subplot(3, 1, 1)
    plt.cla()
    plt.plot(loss_list,"k")
    plt.xlabel("Batches")
    plt.ylabel("Training Loss")
    plt.subplot(3, 1, 2)
    plt.cla()
    plt.plot(correct_list,"r")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    # plt.plot(_predictions_series)
    # plt.plot(batchY)

    for batch_series_idx in range(batch_size):
        # one_hot_output_series = np.array(predictions_series)[:, batch_series_idx]
        single_output_series = np.array([(1 if out > 0.5 else 0) for out in predictions_series[batch_series_idx]])
        index_output = max(enumerate(single_output_series), key=operator.itemgetter(1))
        index_target = max(enumerate(batchY[batch_series_idx]), key=operator.itemgetter(1))
        if index_output == index_target:
            correct += 1
        # plt.subplot(3, 3, 2)
        # plt.cla()
        # # plt.axis([0, truncated_backprop_length, 0, 2])
        # left_offset = range(2)
        # # plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        # plt.bar(left_offset, batchY[batch_series_idx], width=1, color="red")
        # plt.bar(left_offset, single_output_series*0.5, width=1, color="green")
    correct_list.append(float(correct*100/batch_size))
    print(float(correct*100/batch_size))
    plt.draw()
    plt.pause(0.0001)
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    #
    # for epoch_idx in range(num_epochs):
    #     x, y = generateDataMutate()
    #     _current_state = np.zeros((batch_size, state_size))
    #
    #     print("New data, epoch", epoch_idx)
    #
    #     for batch_idx in range(num_batches):
    #         start_idx = random.randint(0, len(x)-1)
    #
    #         # print(start_idx)
    #         r = random.random()
    #         random.shuffle(x[start_idx], lambda: r)
    #         random.shuffle(y[start_idx], lambda: r)
    #         X = np.asarray(x[start_idx][0:batch_size])
    #         batchY = y[start_idx][0:batch_size]
    #         if(X[0][0].shape != X[0][1].shape):
    #             print("Shape mismatch ", X[0][0].shape, " ", X[0][1].shape)
    #         batchX = np.asarray([np.concatenate((A[0], A[1]), axis=1) for A in X])
    #         # print("batchX shape", batchX.shape)
    #         _total_loss, _train_step, _current_state, _predictions_series = sess.run(
    #             [total_loss, train_step, current_state, predictions_series],
    #             feed_dict={
    #                 batchX_placeholder: batchX,
    #                 batchY_placeholder: batchY,
    #                 init_state: _current_state
    #             })
    #
    #         loss_list.append(_total_loss)
    #
    #         if batch_idx%50 == 0:
    #             print("Step",batch_idx, "Loss", _total_loss)
    #             plot(loss_list, _predictions_series, batchX, batchY)
    #             start_idx = random.randint(0, len(x) - 1)
    #
    #             # print(start_idx)
    #             r = random.random()
    #             random.shuffle(x[start_idx], lambda: r)
    #             random.shuffle(y[start_idx], lambda: r)
    #             X = np.asarray(x[start_idx][0:batch_size])
    #             batchY1 = y[start_idx][0:batch_size]
    #             if (X[0][0].shape != X[0][1].shape):
    #                 print("Shape mismatch ", X[0][0].shape, " ", X[0][1].shape)
    #             batchX1 = np.asarray([np.concatenate((A[0], A[1]), axis=1) for A in X])
    #
    #             accuracy = sess.run([accuracy_op], feed_dict={batchX_placeholder: batchX1, batchY_placeholder: batchY1,
    #                                                           init_state: _current_state})
    #             accuracy_list.append(float(accuracy[0]*100))
    #             plt.subplot(3, 1, 3)
    #             plt.cla()
    #             plt.plot(accuracy_list,"b")
    #             plt.xlabel("Epochs")
    #             plt.ylabel("Validation Accuracy")
    #save_path = saver.save(sess, "testResults/model.ckpt")
    saver.restore(sess, "testResults/model.ckpt")
    graph = tf.get_default_graph()
    x, y = readRealData()
    # x, y = generateDataMutate()
    _current_state = np.zeros((batch_size, state_size))
    start_time = time.time()

    for i in range(0, 90000):
        X1 = []
        flag = False
        # x, y = generateDataMutate()
        start_idx = random.randint(0, len(x) - batch_size-1)
        # r = random.random()
        # random.shuffle(x[start_idx], lambda: r)
        # random.shuffle(y[start_idx], lambda: r)
        X = np.asarray(x[start_idx:start_idx+batch_size])
        batchY1 = y[start_idx:start_idx+batch_size]
        for A in X:
            if (A[0].shape != A[1].shape):
                print("Shape mismatch ", A[0].shape, " ", A[1].shape)
                flag = True
                continue
            X1.append(np.concatenate((A[0], A[1]), axis=1))
        if flag is True:
            break
        batchX1 = np.asarray(X1)

        pred = sess.run([predictions_series], feed_dict={batchX_placeholder: batchX1, batchY_placeholder: batchY1,
                                                      init_state: _current_state})
        accuracy = sess.run([accuracy_op], feed_dict={batchX_placeholder: batchX1, batchY_placeholder: batchY1,
                                                      init_state: _current_state})
        test_accuracy_list.append(float(accuracy[0] * 100))
        target_list.append(batchY1)
        pred_list.append(pred)
    # Compute confusion matrix
    pred_list = np.asarray(pred_list)
    pred_list = np.reshape(pred_list, (-1, batch_size, 2))
    pred_list_max = np.argmax(pred_list, axis=2)
    target_list_max = np.argmax(target_list, axis=2)
    cnf_matrix = confusion_matrix(np.reshape(target_list_max,(-1,)),np.reshape(pred_list_max,(-1,)))
    np.set_printoptions(precision=2)
    report = classification_report(np.reshape(target_list_max,(-1,)),np.reshape(pred_list_max,(-1,)))
    print(report)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Matched', 'Unmatched'],
                          title='Confusion matrix, without normalization')

    # print("Time taken For Test ", time.time()-start_time)
    # print(test_accuracy_list)
    # plt.plot(test_accuracy_list, "g")
    # plt.xlabel("Batches")
    # plt.ylabel("Test Accuracy")
    # plt.show()
    # print("Mean Test Accuracy ", np.mean(np.asarray(test_accuracy_list)))
    tn, fp, fn, tp = cnf_matrix.ravel()
    print ("Accuracy ", (tp+tn)/(tn+fp+fn+tp))
    print("Precision ", (tp/(tp +fp)))
    print("Recall ", (tp/(tp+fn)))
plt.ioff()
plt.show()
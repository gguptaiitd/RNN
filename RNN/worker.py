from multiprocessing import Queue, Process
import torch
import numpy as np
import torch.utils.data as data_utils
import os
import netbuilder
import random
from keras.utils import to_categorical
from Bio import SeqIO
series_length = 15
refSeries = 4000
mutSeries = 10
unmatched = mutSeries +1
mutSeries_length = int(series_length*0.4)
unmatched_seq_length = mutSeries + 1
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
        ref_seq = r.seq

    ref_series = {}
    mut_series = mdict()
    loc = random.sample(list(range(0, len(ref_seq)-series_length-1)), refSeries)
    for loc in loc:
        l_seq = series_length
        seq = list(ref_seq[loc:loc + l_seq])
        if (str(seq)).find('N') != -1:
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
        del s_seed
        del s_rlen
    keys = list(ref_series.keys())
    random.shuffle(keys)
    # input_X = []
    batchX = []
    y = []
    for loc in keys:
        for i in range(mutSeries):
            # x.append([ref_series[loc], mut_series[loc][i]])
            refX = np.asarray(ref_series[loc], dtype=np.int32)
            refY = np.asarray(mut_series[loc][i], dtype=np.int32)
            batchX.append(np.asarray([np.concatenate((A, B), axis=0) for A, B in zip(refX, refY)]))
            y.append([1, 0])
        # y= np.ones((mutSeries_length))
        for i in range(unmatched):
            loc_key = keys[:]
            loc_key.remove(loc)
            w_loc = random.choice(loc_key)
            # x.append([ref_series[loc], ref_series[w_loc]])
            refX = np.asarray(ref_series[loc], dtype=np.int32)
            refY = np.asarray(ref_series[w_loc], dtype=np.int32)
            batchX.append(np.asarray([np.concatenate((A, B), axis=0) for A, B in zip(refX, refY)]))
            y.append([0, 1])
    r = random.random()
    random.shuffle(y, lambda: r)
    random.shuffle(batchX, lambda: r)

    return batchX, y


# helper class for scheduling workers
class Scheduler:
    def __init__(self, workerids, use_cuda):
        self._queue = Queue()
        self.workerids = workerids
        self._results = Queue()
        self.use_cuda = use_cuda

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for wid in self.workerids:
            self._workers.append(CustomWorker(wid, self._queue, self._results, self.use_cuda))

    def start(self, xlist):

        # put all of models into queue
        for model_info in xlist:
            self._queue.put(model_info)

        # add a None into queue to indicate the end of task
        self._queue.put(None)

        # start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()

        print("All workers are done")
        returns = []
        networks = []
        for i in range(len(xlist)):
            score, net = self._results.get()
            returns.append(score)
            networks.append(net)

        return networks, returns


class CustomWorker(Process):
    def __init__(self, workerid, queue, resultq, use_cuda):
        Process.__init__(self, name='ModelProcessor')
        self.workerid = workerid
        self.queue = queue
        self.resultq = resultq
        self.use_cuda = use_cuda
        self.x, self.y = generateData()

        batch_size = 10
        train_data = int(len(self.x) * 0.8)
        # traindata = {'X': torch.from_numpy(np.asarray(self.x[0:train_data])),
        #              'Y': torch.from_numpy(np.asarray(self.y[0:train_data]))}
        try:
            traindata = data_utils.TensorDataset(torch.from_numpy(np.asarray(self.x[0:train_data], dtype=np.float32)).double(),
                                             torch.from_numpy(np.asarray(self.y[0:train_data], dtype=np.float32)).double())
            dataloader = data_utils.DataLoader(traindata, batch_size=10, shuffle=True)
            self.train_loader = dataloader
            # testdata = {'X': torch.from_numpy(np.asarray(self.x[train_data:])),
            #             'Y': torch.from_numpy(np.asarray(self.y[train_data:]))}
            testdata = data_utils.TensorDataset(torch.from_numpy(np.asarray(self.x[train_data:], dtype=np.float32)).double(),
                                                 torch.from_numpy(np.asarray(self.y[train_data:], dtype=np.float32)).double())
            testdataset = data_utils.DataLoader(testdata, batch_size=10, shuffle=True)
            self.test_loader = testdataset
        except ValueError:
            # print("train data ", traindata.shape)
            # print("test data ", testdata.shape)
            print ("data ",len(self.x), " ", len(self.y))

    def run(self):
        import rnnNeuro
        while True:
            net = self.queue.get()
            if net == None:
                self.queue.put(None)  # for other workers to consume it.
                break
            # net = net_builder.randomize_network(bounded=False)
            xnet = rnnNeuro.CustomModel(net, self.use_cuda)
            # printing network
            print(xnet)
            ret = xnet.train(self.train_loader)
            score = -1
            if ret == 1:
                score = xnet.test(self.test_loader)
                print('worker_{} score:{}'.format(self.workerid, score))
            self.resultq.put((score, net))
            del xnet

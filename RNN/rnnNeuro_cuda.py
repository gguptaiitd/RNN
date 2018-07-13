import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import random
from torchvision import datasets, transforms
import numpy as np
batch_size = 10
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """forward pass."""
        return x.view(x.size(0), -1)
softmax = nn.Softmax()
epochs = 5
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias, dropout, bidirectional):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bias, False, dropout, bidirectional)
        self.fc = nn.Linear(hidden_size*2, 2)

    def forward(self, input,cuda):
        input = input.view(-1, batch_size, 8)
        # Set initial states
        if self.bidirectional is True:
            h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        if cuda is True:
            h0 = h0.cuda()
        # Forward propagate RNN
        out, hn = self.rnn(input, h0)
        output = self.fc(out)
        output = output.view(batch_size, -1, 2)
        # Decode hidden state of last time step
        output = torch.sum(torch.stack(output), dim=1)
        return softmax(output)


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias, dropout, bidirectional):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, False,dropout, bidirectional)
        self.fc = nn.Linear(hidden_size*2, 2)

    def forward(self, input, cuda):
        input = input.view(-1, batch_size, 8)
        # Set initial states
        if self.bidirectional is False:
            h0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        else:
            h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if cuda is True:
            h0 = h0.cuda()
            c0 = c0.cuda()

        # Forward propagate RNN
        out, hn = self.lstm(input, (h0, c0))
        # Decode hidden state of last time step
        output = self.fc(out)
        output = output.view(batch_size, -1, 2)
        # Decode hidden state of last time step
        output = torch.sum(torch.stack(output), dim=1)
        return softmax(output)

class CustomModel():

    def __init__(self, build_info, CUDA=True):

        previous_units = 8
        self.model = nn.Sequential()
        # self.model.add_module('flatten', Flatten())
        for i, layer_info in enumerate(build_info['layers']):
            # i = str(i)
            # self.model.add_module(
            #     'fc_' + i, nn.Linear(previous_units, layer_info['input_size']['val']))
            if build_info['type']['val'] == 'RNN':
                self.model.add_module('DNN ', RNN(previous_units, layer_info['hidden_size']['val'],\
                                                  build_info['num_layers']['val'], layer_info['bias']['val'], \
                                                  layer_info['dropout_rate']['val'], layer_info['bidirectional']['val']))
            elif build_info['type']['val'] == 'LSTM':
                self.model.add_module('DNN ', LSTM(previous_units, layer_info['hidden_size']['val'], \
                                                   build_info['num_layers']['val'], layer_info['bias']['val'], \
                                                  layer_info['dropout_rate']['val'],layer_info['bidirectional']['val']))
        self.model.add_module('sofmax', nn.Softmax())
        self.model.cpu()

        if build_info['optimizer']['val'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=build_info['weight_decay']['val'],
                                   weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'adadelta':
            optimizer = optim.Adadelta(self.model.parameters(),
                                       lr=build_info['weight_decay']['val'],
                                       weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(),
                                       lr=build_info['weight_decay']['val'],
                                       weight_decay=build_info['weight_decay']['val'])

        elif build_info['optimizer']['val'] == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(),
                                      lr=build_info['weight_decay']['val'],
                                      weight_decay=build_info['weight_decay']['val'])
        else:
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=build_info['weight_decay']['val'],
                                  weight_decay=build_info['weight_decay']['val'],
                                  momentum=0.9)
        self.optimizer = optimizer
        self.cuda = False
        if CUDA:
            self.model.cuda()
            self.cuda = True

    #
    # def forward(self, input):
    #     input = input.view(-1)
    #     x = self.input(self.flat(input))
    #     linear_out = x.view(1, -1, 8)
    #     net_out = self.net(linear_out)
    #     softmax_out = self.hidden_output(self.net(net_out))
    #     out = self.output(softmax_out)
    #     return out

    def train(self, train_loader, max_batches=100):
        """Train for 1 epoch."""
        self.model.train()

        batch = 0
        for epoch_idx in range(epochs):
            for batch_idx, (dataSet, targetSet) in enumerate(train_loader):
                if self.cuda:
                    dataSet, targetSet = dataSet.cuda(), targetSet.cuda()
                _data = Variable(dataSet.view(batch_size, 1, -1, 8)).float()
                _target = Variable(targetSet.view(1, batch_size, 2)).float()
                self.optimizer.zero_grad()
                output = self.model(_data, self.cuda)
                _target = _target.view(batch_size, 2)
                loss = F.binary_cross_entropy_with_logits(output, _target)
                np_loss = loss.cpu().data.numpy()
                # print((np_loss))
                if np.isnan(np_loss):
                    print('stopping training - nan loss')
                    return -1
                elif loss.cpu().data.numpy()[0] > 100000:
                    print('Qutting, loss too high', np_loss)
                    return -1
                pred = softmax(output).data.max(1)[1]
                correct = pred.eq(_target.data.max(1)[1]).cpu().sum()
                accuarcy = 100. * correct / batch_size
                # print("Accuracy ", accuarcy)
                loss.backward()
                self.optimizer.step()
                # batch += 1
                # if batch > max_batches:
                #     break
        return 1

    def test(self, test_loader, CUDA=False):
        """Evaluate a model."""
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            _data = Variable(data.view(10, 1, -1, 8)).float()
            _target = Variable(target.view(1, 10, 2)).float()
            output = self.model(_data, self.cuda)
            _target = _target.view(batch_size, 2)
            test_loss += F.binary_cross_entropy_with_logits(output, _target).data[0]
            pred = softmax(output).data.max(1)[1]
            correct += pred.eq(_target.data.max(1)[1]).cpu().sum()

        test_loss /= len(test_loader)
        accuarcy = 100. * correct / len(test_loader.dataset)
        print ("Accuracy ",accuarcy )
        return accuarcy

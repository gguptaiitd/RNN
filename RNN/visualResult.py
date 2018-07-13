import pickle
import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd

stats_net = pickle.load(open('testResults/Run_rnn_5.pop', 'rb'))
stats = pickle.load(open('testResults/Run_rnn_5.stats', 'rb'))
print (len(stats[0]))
stats = np.array(stats)

x = []
y = []
mu = []
print(stats.shape)
for e in range(stats.shape[0]):
    x += [e] * stats.shape[1]
    y += list(stats[e])
    mu.append(np.mean(y))
data = pd.DataFrame({'epoch':x, 'accuracy':y})

mus = pd.DataFrame({'epoch':np.arange(stats.shape[0]), 'accuracy':mu})
print(mus)

max_i = np.argmax(y)
x[max_i], y[max_i]
pl.figure(figsize=(8,8))
sns.boxplot(x="epoch", y="accuracy", data=data)
# sns.swarmplot(x="epoch", y="accuracy", data=data)
sns.stripplot(x="epoch", y="accuracy", data=mus, jitter=True, size=8, color=".1", linewidth=0)
# label = 'Max accuracy:{}%'.format(y[max_i])
# pl.plot([y[max_i]]*10, label=label)
pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

print(y[max_i])


sns.distplot(y, bins=20, kde=False, rug=False)

max_i = np.argmax(y)
x[max_i], y[max_i]
pl.figure(figsize=(8,8))
# sns.boxplot(x="epoch", y="accuracy", data=data)
sns.swarmplot(x="epoch", y="accuracy", data=data)
sns.stripplot(x="epoch", y="accuracy", data=mus, jitter=True, size=8, color=".1", linewidth=0)
label = 'Max accuracy:{}%'.format(y[max_i])
pl.plot([y[max_i]]*10, label=label)
pl.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
print(y[max_i])
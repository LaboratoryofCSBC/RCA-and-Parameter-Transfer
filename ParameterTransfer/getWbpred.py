from msvr import kernelmatrix
from msvr import msvr
import numpy as np
from data import data
from findclass import dele
import datetime
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--nclass', type=int, default=4,
                    help='num of classes (default: 4)')
parser.add_argument('--n_feature', type=int, default=253,
                    help='feature dimension (default: 253)')
parser.add_argument('--tar_n', type=int, default=1,
                    help='target subject (default: 1)')
args = parser.parse_args()


W_all = np.loadtxt('./msvr/msvr_W.txt', delimiter=',')
b_all = np.loadtxt('./msvr/msvr_b.txt', delimiter=',')
x_src = np.loadtxt('./msvr/msvr_x.txt', delimiter=',')


# Delete the W and b obtained from the target subject, and keep only the W and b obtained from the source domain.
tar_dele = dele(args.tar_n)
W_all = np.delete(W_all, tar_dele, axis=0)
b_all = np.delete(b_all, tar_dele, axis=0)
x_src = np.delete(x_src, tar_dele, axis=0)

# Read target domain data
n = np.size(W_all, 1)# num of samples
x_tar, y_tar, _ = data(args.tar_n)

# Input & Output
# Xtrain: number of samples * input dimension
# Ytrain: number of samples * output dimension
Xtest = np.reshape(x_tar, [1, -1])
Xtrain = x_src
Ytrain = np.hstack((W_all, b_all))

print('Xtest:', Xtest.shape)
print('Xtrain:', Xtrain.shape)
print('Ytrain:', Ytrain.shape)

# Parameters
#  ker: kernel ('lin', 'poly', 'rbf'),
#  C: cost parameter,
#  par (kernel):
#	  -lin: no parameters,
#	  -poly: [gamma, b, degree],
#	  -rbf: sigma (width of the RBF kernel),
#  tol: tolerance.

ker  = 'RBF'
C    = 2
epsi = 0.001
par  = 0.8 # if kernel is 'rbf', par means sigma
tol  = 1e-10

# Train
Beta = msvr(Xtrain, Ytrain, ker, C, epsi, par, tol)
# print('Beta:', Beta.shape)

# Predict with train set
H = kernelmatrix('RBF', Xtrain, Xtrain, par);
Ypred = np.dot(H, Beta)
# print('train', Ypred)

# Predict with test set
H = kernelmatrix('RBF', Xtest, Xtrain, par);
Ypred = np.dot(H, Beta)
# print('test', Ypred.shape)
Wpred = Ypred[:, 0:n]
Wpred = np.reshape(Wpred, [args.n_feature, args.nclass])
# print(Wpred.shape)
bpred = Ypred[:, n:]
# print(bpred)
np.savetxt("./msvr/Wpred_" + str(args.tar_n) + ".txt", Wpred, fmt='%f', delimiter=',')
np.savetxt("./msvr/bpred_" + str(args.tar_n) + ".txt", bpred, fmt='%f', delimiter=',')






# Here are the TPT tests
starttime = datetime.datetime.now()
x_test = np.loadtxt('./sample/x_test_' + str(args.tar_n) + '.txt', delimiter=',')
y_test = np.loadtxt('./sample/y_test_' + str(args.tar_n) + '.txt', delimiter=',')
nx = np.size(x_test, 0)
arr = np.dot(x_test, Wpred)+bpred
arr_max = arr.max(axis=1).reshape(nx, 1)
arr = arr - arr_max
arr_exp = np.exp(arr)
arr_sum = arr_exp.sum(axis=1).reshape(nx, 1)
arr_softmax = arr_exp / arr_sum
y_p = arr_softmax
index = np.arange(0, nx)
correct = index[np.argmax(y_p, 1) == np.argmax(y_test, 1)]
correct = np.reshape(correct, [-1, 1])
# print(correct.shape)
[m, n] = np.shape(correct)
# print(m)
accuracy = m/nx
print('TPT_accuracy:')
print(accuracy)
endtime = datetime.datetime.now()
seconds = (endtime - starttime)
print('The running time is', seconds)
import tensorflow as tf
import numpy as np
import datetime
from findclass import data_class
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--nclass', type=int, default=4,
                    help='num of classes (default: 4)')
parser.add_argument('--n_feature', type=int, default=253,
                    help='feature dimension (default: 253)')
parser.add_argument('--tar_n', type=int, default=1,
                    help='target subject (default: 1)')
parser.add_argument('--epoch', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--n_class', type=int, default=10,
                    help='number of labeled samples available for each class in the target domain,'
                         ' less than 1/4 of the total, 10 < 72/4 (default: 10)')
args = parser.parse_args()

Wpred = np.loadtxt('./msvr/Wpred_' + str(args.tar_n) + '.txt', delimiter=',')
bpred = np.loadtxt('./msvr/bpred_' + str(args.tar_n) + '.txt', delimiter=',')
Wpred = np.reshape(Wpred, [args.n_feature, args.nclass])
bpred = np.reshape(bpred, [1, args.nclass])
print()
print('starting SPT_test')

x_test = np.loadtxt('./sample/x_test_' + str(args.tar_n) + '.txt', delimiter=',')
y_test = np.loadtxt('./sample/y_test_' + str(args.tar_n) + '.txt', delimiter=',')
train_x = np.loadtxt('./sample/train_x_' + str(args.tar_n) + '.txt', delimiter=',')
train_y = np.loadtxt('./sample/train_y_' + str(args.tar_n) + '.txt', delimiter=',')
x_train, y_train, _, _, _ = data_class(train_x, train_y, args.n_class)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, args.n_feature])
ys = tf.placeholder(tf.float32, [None, args.nclass])
keep_prob = tf.placeholder(tf.float32)

# fc1 layer ##
W_fc1 = weight_variable([args.n_feature, args.nclass])
b_fc1 = bias_variable([args.nclass])
prediction = tf.nn.softmax(tf.matmul(xs, W_fc1) + b_fc1)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
cross_msvr = tf.reduce_sum(tf.square(W_fc1-Wpred), 1) + tf.reduce_sum(tf.square(b_fc1-bpred), 1)
cross = cross_entropy + 0.3*cross_msvr
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross)


sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
#     init = tf.initialize_all_variables()
# else:
#     init = tf.global_variables_initializer()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

for i in range(args.epoch):
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
    train_loss = sess.run(cross_entropy, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})

    if i % 50 == 0:
        print('time:', i)
        print('train_loss:', train_loss)

test_output = compute_accuracy(x_test, y_test)
saver.save(sess, "save/SPT/model" + str(args.tar_n) + ".ckpt")

sess.close()
print('tar_subject:', args.tar_n)
print('LabeledSamplefromTarSub:', args.n_class)
print('SPT_accuracy:', test_output)


'''
starttime = datetime.datetime.now()
saver.restore(sess, "save/SPT/model" + str(args.tar_n) + ".ckpt")
# y_pre = sess.run(prediction, feed_dict={xs: x_test, keep_prob: 1})
# print(y_pre)
print(compute_accuracy(x_test, y_test))
endtime = datetime.datetime.now()
seconds = (endtime - starttime)
print('The running time is', seconds)
'''
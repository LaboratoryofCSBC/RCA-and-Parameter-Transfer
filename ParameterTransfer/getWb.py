import tensorflow as tf
import numpy as np
from findclass import data_class
from data import data
import argparse

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

W_all = None
b_all = None
x_src = None

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--n_subject', type=int, default=5,
                    help='Total number of subjects = number of subjects in source domain + number of subjects '
                         'in target domain (default: 5)')
parser.add_argument('--n_class', type=int, default=18,
                    help='For each category, take 1/4 of the total number of samples, 18 = 72/4 (default: 18)')
parser.add_argument('--nclass', type=int, default=4,
                    help='num of classes (default: 4)')
parser.add_argument('--n_feature', type=int, default=253,
                    help='feature dimension (default: 253)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='target subject (default: 1)')
args = parser.parse_args()


file = open('./msvr/CrossValidation_accuracy.txt', 'w')
n_select = 0
for n_sub in range(args.n_subject):
    # tf.reset_default_graph()
    n_select = n_select + 1
    x, y, subject = data(n_select)

    p = np.random.permutation(x.shape[0])
    x = x[p, :]
    y = y[p, :]
    # 4 - Cross-validation
    x_1, y_1, x_remain, y_remain, number_1 = data_class(x, y, args.n_class)
    x_2, y_2, x_remain, y_remain, number_2 = data_class(x_remain, y_remain, args.n_class)
    x_3, y_3, x_remain, y_remain, number_3 = data_class(x_remain, y_remain, args.n_class)
    x_4, y_4, x_remain, y_remain, number_4 = data_class(x_remain, y_remain, args.n_class)

    number = [number_1, number_2, number_3, number_4]

    for j in range(4):
        tf.reset_default_graph()
        print()
        print('subject_', subject, 'The', j+1, 'th:')
        x_train = np.delete(x, number[j], axis=0)
        y_train = np.delete(y, number[j], axis=0)
        x_test = x[number[j]]
        y_test = y[number[j]]


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
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

        sess = tf.Session()
        # important step
        init = tf.global_variables_initializer()
        sess.run(init)

        # saver = tf.train.Saver(max_to_keep=1)
        max_acc = 0.25
        acc = 0
        for i in range(args.epochs):
            sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
            train_loss = sess.run(cross_entropy, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})

            test_output = compute_accuracy(x_test, y_test)
            if acc < test_output:
                acc = test_output
                W = sess.run(W_fc1)
                b = sess.run(b_fc1)

            if i % 50 == 0:
                print('time:', i)
                print('train_loss:', train_loss)
                print('train:', compute_accuracy(x_train, y_train), 'test:', test_output)

        sess.close()
        print('subject_', subject, 'Accuracy of the', j + 1, 'th:', acc)
        out1 = str(subject)
        out2 = str(j+1)
        string = '\nsubject_' + out1 + 'Accuracy of the' + out2 + 'th:'
        # print(string)
        file.write(string)
        file.write(format(acc, '.4f'))

        W = np.reshape(W, [1, -1])
        b = np.reshape(b, [1, -1])
        x_s = np.reshape(x, [1, -1])
        # x_s = np.reshape(x_train, [1, -1])
        if W_all is None:
            W_all = W
        else:
            W_all = np.vstack((W_all, W))
        if b_all is None:
            b_all = b
        else:
            b_all = np.vstack((b_all, b))
        if x_src is None:
            x_src = x_s
        else:
            x_src = np.vstack((x_src, x_s))
        print(W_all.shape)
        print(b_all.shape)
        print(x_src.shape)

np.savetxt("./msvr/msvr_W.txt", W_all, fmt='%f', delimiter=',')
np.savetxt("./msvr/msvr_b.txt", b_all, fmt='%f', delimiter=',')
np.savetxt("./msvr/msvr_x.txt", x_src, fmt='%f', delimiter=',')







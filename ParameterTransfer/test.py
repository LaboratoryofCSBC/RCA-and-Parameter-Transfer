import tensorflow as tf
from SPT import x_train, y_train, x_test, y_test, args
import datetime

print()
print('starting test')

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
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

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
saver.save(sess, "save/Softmax/model" + str(args.tar_n) + ".ckpt")

sess.close()
print('tar_subject:', args.tar_n)
print('LabeledSamplefromTarSub:', args.n_class)
print('Softmax_accuracy:', test_output)

'''
starttime = datetime.datetime.now()
saver.restore(sess, "save/Softmax/model" + str(args.tar_n) + ".ckpt")
# y_pre = sess.run(prediction, feed_dict={xs: x_test, keep_prob: 1})
# print(y_pre)
print(compute_accuracy(x_test, y_test))
endtime = datetime.datetime.now()
seconds = (endtime - starttime)
print('The running time is', seconds)
'''
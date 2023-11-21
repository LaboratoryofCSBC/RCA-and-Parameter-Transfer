import numpy as np
from data import data
from findclass import data_class

tar_n = 1   # target subject
n_class = 18    # For each category, take 1/4 of the total number of samples, 18 = 72/4

x_tar, y_tar, _ = data(tar_n)

train_x, train_y, x_test, y_test, _ = data_class(x_tar, y_tar, n_class)
np.savetxt("./sample/train_x_" + str(tar_n) + ".txt", train_x, fmt='%f', delimiter=',')
np.savetxt("./sample/train_y_" + str(tar_n) + ".txt", train_y, fmt='%f', delimiter=',')
np.savetxt("./sample/x_test_" + str(tar_n) + ".txt", x_test, fmt='%f', delimiter=',')
np.savetxt("./sample/y_test_" + str(tar_n) + ".txt", y_test, fmt='%f', delimiter=',')
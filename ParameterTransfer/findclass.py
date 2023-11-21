import numpy as np
import random
# Applies to 4 classifications, other category numbers modified as needed

def find_class(label):
    class_1 = []
    class_2 = []
    class_3 = []
    class_4 = []
    n = np.size(label, 0)  # num of samples
    for i in range(n):
        if np.nonzero(label[i])[0] == 0:
            class_1.append(i)
        if np.nonzero(label[i])[0] == 1:
            class_2.append(i)
        if np.nonzero(label[i])[0] == 2:
            class_3.append(i)
        if np.nonzero(label[i])[0] == 3:
            class_4.append(i)
    return class_1, class_2, class_3, class_4

def data_class(data, label, n):
    class_1, class_2, class_3, class_4 = find_class(label)
# Select n samples from each category and obtain the subscripts of these samples.
    class1 = random.sample(class_1, n)
    class2 = random.sample(class_2, n)
    class3 = random.sample(class_3, n)
    class4 = random.sample(class_4, n)

    number_class = np.hstack((class1, class2))
    number_class = np.hstack((number_class, class3))
    number_class = np.hstack((number_class, class4))

    data_train = data[number_class]
    label_train = label[number_class]

    # Randomly disrupt the combined samples
    per = np.random.permutation(data_train.shape[0])
    data_train = data_train[per, :]
    label_train = label_train[per, :]

    data_test = np.delete(data, number_class, axis=0)
    label_test = np.delete(label, number_class, axis=0)

    return data_train, label_train, data_test, label_test, number_class


def dele(n):
    x = [x for x in range(4 * (n - 1), 4 * n)]  # 4 - Cross-validation
    return x
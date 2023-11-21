from scipy.io import loadmat as load

path = './vector/'
def data(n_select):
    data = load(path + 'vector_' + str(n_select))
    x = data['input']
    data = load(path + 'label_' + str(n_select))
    y = data['label']
    subject = str(n_select)
    return x, y, subject
import pickle

def read_pickle(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
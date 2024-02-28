import numpy as np

def generate_context():
    a = np.reshape(np.tile([1, 3-1, 0.913 - 0.5], 23), (23, 3))
    b = np.reshape(np.tile([1, 3-1, 0.818 - 0.5], 11), (11, 3))
    c = np.reshape(np.tile([1, 2-1, 0.906 - 0.5], 32), (32, 3))
    d = np.reshape(np.tile([0, 3-1, 0.727 - 0.5], 11), (11, 3))
    e = np.reshape(np.tile([1, 2-1, 0.800 - 0.5], 15), (15, 3))
    f = np.reshape(np.tile([1, 2-1, 0.500 - 0.5], 8), (8, 3))
    g = np.reshape(np.tile([1, 1-1, 0.913 - 0.5], 23), (23, 3))
    h = np.reshape(np.tile([0, 3-1, 0.913 - 0.5], 23), (23, 3))
    i = np.reshape(np.tile([0, 2-1, 0.500 - 0.5], 8), (8, 3))
    j = np.reshape(np.tile([0, 2-1, 0.800 - 0.5], 15), (15, 3))
    k = np.reshape(np.tile([1, 1-1, 0.546 - 0.5], 11), (11, 3))
    l = np.reshape(np.tile([0, 2-1, 0.937 - 0.5], 32), (32, 3))
    m = np.reshape(np.tile([0, 1-1, 0.546 - 0.5], 11), (11, 3))
    n = np.reshape(np.tile([0, 1-1, 0.913 - 0.5], 23), (23, 3))
    context = np.vstack([a, b, c, d, e, f, g, h, i, j, k, l, m, n])
    np.random.shuffle(context)
    return context
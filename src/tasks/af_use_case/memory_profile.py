import numpy as np
import time
from sklearn import tree
from memory_profiler import profile

@profile
def memory_test(n):
    X = np.stack([np.random.randn(15) for i in range(n)])
    y = np.stack([np.random.binomial(1, 0.3) for i in range(n)])
    clf = tree.DecisionTreeClassifier().fit(X, y)


if __name__ == '__main__':
    t0, n = time.time(), int(1e6)
    memory_test(n)
    print(f'Tree fitted on {n} row took {time.time() - t0}')

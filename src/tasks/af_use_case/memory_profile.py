import numpy as np
import time
from sklearn import tree
from memory_profiler import profile

@profile
def memory_test(l_features, l_labels):
    X, y = np.array(l_features), np.array(l_labels)
    clf = tree.DecisionTreeClassifier().fit(X, y)


if __name__ == '__main__':
    n = int(500000)
    l_features = [list(np.random.randn(15)) for i in range(n)]
    l_labels = [np.random.binomial(1, 0.3) for i in range(n)]
    t0 = time.time()
    memory_test(l_features, l_labels)
    print(f'Tree fitted on {n} row took {time.time() - t0}')

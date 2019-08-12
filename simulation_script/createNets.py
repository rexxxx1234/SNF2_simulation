import pickle
import copy

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Import datasets.
x1 = pd.read_csv('lab_pre/case6/butterfly_1.csv', header=None)
x2 = pd.read_csv('lab_pre/case6/butterfly_2.csv', header=None)
#x3 = pd.read_csv('butterfly_3.csv', header=None)
labels = pd.read_csv('lab_pre/case6/butterfly_labels.csv', header=None).T.values[0]

# Create dictionary of label to index.
label_dict = {}
for i, label in enumerate(labels):
    if label not in label_dict:
        label_dict[label] = []
    label_dict[label].append(i)
    

def extend_data(df, label_dict, n_replicates=1000):
    """
    For each class, create new observations by sampling an interpolation
    of other points in that class.
    """

    X = df.values

    # Copy 'X' to be augmented by additional replicate points.
    X_rep = X.copy()

    # Copy 'label_dict' to be augmented by additional points.
    label_dict_rep = copy.deepcopy(label_dict)

    for label, idxs in label_dict.items():

        # List to store new replicates.
        new_reps = []

        for rep in range(int(n_replicates / len(label_dict.keys()))):

            # Get a random interpolation between two random points 
            # in X, subsetted by idx.
            weights = np.random.uniform(size=2).reshape((-1, 1))
            weights = weights / np.sum(weights)
            rand_idxs = np.random.choice(idxs, size=2)
            X_interp = np.sum(weights * X[rand_idxs], axis=0)

            new_reps.append(X_interp)

        new_reps = np.array(new_reps)
        label_dict_rep[label] += list(range(X_rep.shape[0], X_rep.shape[0] + len(new_reps)))
        X_rep = np.vstack((X_rep, new_reps))

    return X_rep, label_dict_rep


def add_noise(X, std=0.05):
    """
    Adds noise to the input feature matrix 'X'.
    """

    X += np.random.normal(loc=0, scale=std, size=X.shape)

    return X


def create_net(X, perc=98):

    # Get cosine distance matrix and convert to similarity.
    X_dist = squareform(pdist(X, metric='cosine'))
    X_sim = np.max(X_dist) - X_dist

    percentiles = np.percentile(X_sim, perc, axis=1)
    X_sim[X_sim < percentiles] = 0

    X_sim = (X_sim + X_sim.T) / 2

    return X_sim


def make_df(x, X1, X2, labels_1, labels_2):
    """
    Labels rows in the generated datasets X1 and X2.
    Here 'x' corresponds to one of the original datasets.
    """

    shared_idx = list(range(x.shape[0]))

    X1_labels = shared_idx + list(range(len(shared_idx), X1.shape[0]))
    X2_labels = shared_idx + list(range(X1.shape[0], X1.shape[0] - len(shared_idx) + X2.shape[0]))

    X1_labels = ['butterfly_{}'.format(x) for x in X1_labels]
    X2_labels = ['butterfly_{}'.format(x) for x in X2_labels]

    X1_df = pd.DataFrame(X1, index=X1_labels, columns=X1_labels)
    X2_df = pd.DataFrame(X2, index=X2_labels, columns=X2_labels)

    def make_new_dict(X, labels):
        """
        Replaces label indices with actual index names in 'X'.
        """

        new_labels = {}

        for key, val in labels.items():
            if key not in new_labels:
                new_labels[key] = []
            for v in val:
                new_labels[key].append(X.index[v])

        return new_labels

    new_labels_1 = make_new_dict(X1_df, labels_1)
    new_labels_2 = make_new_dict(X2_df, labels_2)

    # Get a dictionary containing all cluster, data point assignments.
    total_labels = {}
    for key, val1, val2 in zip(new_labels_1.keys(), new_labels_1.values(), new_labels_2.values()):
        if key not in total_labels:
            total_labels[key] = []
        added = set()
        for v in val1:
            if v not in added:
                added.add(v)
                total_labels[key].append(v)
        for v in val2:
            if v not in added:
                added.add(v)
                total_labels[key].append(v)

    # Invert dictionary mapping, so it's cluster label to observation name.
    inv_total_labels = {}
    for key, val in total_labels.items():
        for v in val:
            inv_total_labels[v] = key

    return X1_df, X2_df, inv_total_labels


X1, labels_1 = extend_data(x1, label_dict, n_replicates=250)
X1 = add_noise(X1)
#X1 = create_net(X1)
np.savetxt("/Users/mashihao/Desktop/SNF/simulated_data/lab_pre/case6/X1_ori_case6.csv", X1, delimiter=",")
#np.savetxt("/Users/mashihao/Desktop/SNF/simulated_data/lab_pre/X1_labels_case1.csv", labels_1[1], delimiter=",")

X2, labels_2 = extend_data(x2, label_dict, n_replicates=250)
X2 = add_noise(X2)
#X2 = create_net(X2)
np.savetxt("/Users/mashihao/Desktop/SNF/simulated_data/lab_pre/case6/X2_ori_case6.csv", X2, delimiter=",")
#np.savetxt("/Users/mashihao/Desktop/SNF/simulated_data/lab_pre/X2_labels_case1.csv", labels_2[1], delimiter=",")



'''
X3, labels_3 = extend_data(x3, label_dict, n_replicates=400)
X3 = add_noise(X3)
print(X3.shape)
np.savetxt("/Users/mashihao/Desktop/X3_ori.csv", X3, delimiter=",")
#print(labels_3[1])
np.savetxt("/Users/mashihao/Desktop/W3_labels.csv", labels_3[1], delimiter=",")
'''



'''
X1, X2, labels = make_df(x1, X1, X2, labels_1, labels_2)

X1.to_csv('X1.csv')
X2.to_csv('X2.csv')
pickle.dump(labels, open('all_labels.pickle', 'wb'))
'''

def eval_net():
    """
    """
    pass
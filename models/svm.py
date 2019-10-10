import os
import numpy as np
import copy
from sklearn.svm import SVC
import helpers.bm25

def train(train_set, embeddings, y):
    # embedding_size = embeddings.shape[1]
    
    # class_weight='balanced' ?
    model = SVC(C=0.9, kernel='linear', probability=True, verbose=False)
    model.fit(embeddings[train_set], y[train_set])

    return model

def bootstrap(n=10):
    bm25 = helpers.bm25.compute(
        os.path.join([os.path.dirname(os.path.abspath(__file__)),'../data/Hall.csv']),
        query=['defect', 'prediction'])
    top_args = bm25.argsort()[-n:]
    return top_args.tolist()

def query(classifier, embeddings, pool, relevant, n=1):
    # sampling
    prob = classifier.predict_proba(embeddings[pool])
    # certainty = prob.max(axis=1)
    uncertainty = 1 - prob.max(axis=1)
    x = np.argsort(uncertainty)[-n:]
    return x.tolist()

def include(x, relevant, train_set):
    train_set += x
    # if(id > 8806 (or -104)): positive; else negative 
    relevant += [idx for idx in x if idx > 8806]

x_neg = np.column_stack((
    np.load('../../concat/-1/x_neg.npy'), 
    np.load('../../concat/3/x_neg.npy')))
x_pos = np.column_stack((
    np.load('../../concat/-1/x_pos.npy'), 
    np.load('../../concat/3/x_pos.npy')))
embeddings = np.concatenate((x_neg, x_pos), axis=0)
del x_pos
del x_neg

budget = 50
n_docs = 8911

train_set = []
relevant = []
pool = list(range(0, n_docs)) # all the docs
y = np.array([0]*8807 + [1]*104)

x = bootstrap()
include(x, relevant, train_set)
pool = [doc for doc in pool if doc not in x]
print('recall: {0}%'.format((len(relevant)/n_docs)*100))

for i in range(int(budget) - 1):
    classifier = train(train_set, embeddings, y)
    x = query(classifier, embeddings, pool, relevant)
    include(x, relevant, train_set) # simulate review
    pool = [doc for doc in pool if doc not in x]
    
    print('recall: {0}%'.format((len(relevant)/n_docs)*100))
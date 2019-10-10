import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_bm_25(query, documents):
    b = 0.75
    k1 = 1.5

    content = documents
    tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False, smooth_idf=False,
                              sublinear_tf=False, decode_error="ignore")
    tf = tfidfer.fit_transform(content)

    d_avg = np.mean(np.sum(tf, axis=1))
    score = {}
    for word in query:
        score[word] = []
        try:
            index = tfidfer.vocabulary_[word]
        except:
            score[word] = [0] * len(content)
            continue
        df = sum([1 for wc in tf[:, index] if wc > 0])
        idf = np.log((len(content) - df + 0.5) / (df + 0.5))
        for i in range(len(content)):
            score[word].append(
                idf * tf[i, index] / (tf[i, index] + k1 * ((1 - b) + b * np.sum(tf[0], axis=1)[0, 0] / d_avg)))

    bm = np.sum(list(score.values()), axis=0)
    
    return bm

def compute(path_csv, query=['defect', 'prediction']):
    df = pd.read_csv(path_csv, sep=',', index_col=False)
    documents = [abstract + text for abstract, text in (df['Abstract'].tolist(), df['Document Title'].tolist())]
    bm = compute_bm_25(query, documents)
    return bm
import pandas
import csv
import cPickle
import numpy as np
import random
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures

all_data = pandas.read_csv('goldTruth.txt', sep=';', header=None,
                            quoting=csv.QUOTE_NONE)
vector_map = cPickle.load(open('vectors2.pkl'))

def get_subdirectories(df):
    subdirs = defaultdict(lambda: defaultdict(list))
    for i, row in df.iterrows():
        subdir = row[2].split('_')[0]
        cluster = row[1]
        subdirs[subdir][cluster].append(i)
    return subdirs

def preprocess_vectors():
    # Normalize mean
    global mvec
    mvec = np.mean(np.array(vector_map.values()), axis=0)

def get_pairs(subdir):
    pos_pairs = []
    svals = subdir.values()
    for cluster in svals:
        for a in xrange(len(cluster)):
            for b in xrange(a+1, len(cluster)):
                pos_pairs.append((cluster[a], cluster[b]))

    neg_pairs = []
    for i in xrange(len(subdir)):
        for j in xrange(i+1, len(subdir)):
            c1, c2 = svals[i], svals[j]
            neg_pairs.extend((c1[a], c2[b]) for a in xrange(len(c1))
                for b in xrange(len(c2)))

    return (pos_pairs, neg_pairs)

def normalize_pairs(pairset):
    for (a, b) in list(pairset):
        if (b, a) in pairset:
            pairset.remove((b, a))

    for (a, b) in list(pairset):
        if a > b:
            pairset.remove((a, b))
            pairset.add((b, a))


#Naive implementation, potentially O(n^3) but faster in practice
def agglomerate(clusters):
    curr_clusters = map(set, clusters)
    flag = True
    while flag:
        iters = 0
        flag = False
        n = len(curr_clusters)

        for i in xrange(n):
            for j in xrange(i+1, n):
                iters += 1
                if len(curr_clusters[i] & curr_clusters[j]) > 0:
                    flag = True
                    new_cluster = curr_clusters[i] | curr_clusters[j]
                    curr_clusters.pop(i)
                    curr_clusters.pop(j-1)
                    curr_clusters.append(new_cluster)
                    break
            if flag: break
    return map(list, curr_clusters)

# This one is O(n^2)
def graph_agglomerate(clusters):
    curr_clusters = map(set, clusters)
    n = len(curr_clusters)
    edges = defaultdict(list)

    for i in xrange(n):
        for j in xrange(i+1, n):
            if len(curr_clusters[i] & curr_clusters[j]) > 0:
                edges[i].append(j)
                edges[j].append(i)

    components = np.zeros(n, dtype=np.int)
    cmp = 0

    for i in xrange(n):
        if components[i] == 0:
            cmp += 1
            stack = [i]
            while len(stack):
                curr = stack.pop()
                if components[curr] != 0:
                    continue
                components[curr] = cmp
                for neighbor in edges[curr]:
                    stack.append(neighbor)
    new_clusters = [set() for i in xrange(cmp)]
    for i in xrange(n):
        new_clusters[components[i]-1].update(curr_clusters[i])

    return map(list, new_clusters)

def compute_fscore(pos_pairs, pred_pos):
    map(normalize_pairs, [pos_pairs, pred_pos])

    tp = len(pred_pos & pos_pairs) * 1.0
    fp = len(pred_pos - pos_pairs) * 1.0
    fn = len(pos_pairs - pred_pos) * 1.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)

    print precision, recall
    print "F-Score:", fscore
    print len(pos_pairs), len(pred_pos)

def lemma_baseline():
    subdirs = get_subdirectories(all_data)

    # True data
    pos_pairs = set()
    for subdir in subdirs:
        subdir_num = int(subdir)
        if 20 < subdir_num <= 30:
            pp, _ = get_pairs(subdirs[subdir])
            pos_pairs |= set(pp)

    # Generate Lemma Clusters
    lclusters = defaultdict(lambda: defaultdict(set))
    for i, row in all_data.iterrows():
        subdir = row[2].split('_')[0]
        subdir_num = int(subdir)
        if 20 < subdir_num <= 30:
            lemmas = set(row[5].split())
            for lemma in lemmas:
                lclusters[subdir][lemma].add(i)

    # Merge Lemma Clusters (Naive)
    for subdir in lclusters:
        lclusters[subdir] = dict(enumerate(agglomerate(lclusters[subdir].values())))

    # Predicted data
    pred_pos = set()
    for subdir in lclusters:
        pp, _ = get_pairs(lclusters[subdir])
        pred_pos |= set(pp)

    compute_fscore(pos_pairs, pred_pos)

class WordCache(dict):
    def __missing__(self, key):
        self[key] = set(all_data[5][key].split())
        return self[key]

class VectorCache(dict):
    def __missing__(self, key):
        words = zip(all_data[4][key].split(), all_data[5][key].split())
        word2vec = np.zeros(1000)
        n = len(words)
        for word, lemma in words:
            if word in vector_map:
                word2vec += vector_map[word] - mvec
            elif word.lower() in vector_map:
                word2vec += vector_map[word.lower()] - mvec
            elif lemma in vector_map:
                word2vec += vector_map[lemma] - mvec
            elif lemma.lower() in vector_map:
                word2vec += vector_map[lemma.lower()] - mvec

        if n > 0:
            word2vec /= n
        self[key] = word2vec
        return self[key]

class CharCtrCache(dict):
    def __missing__(self, key):
        self[key] = Counter(all_data[5][key])
        return self[key]

def make_cosine_lookup_table(pairs_to_lookup):
    table = {}
    cache = WordCache()
    for (i, j) in pairs_to_lookup:
        table[(i, j)] = (1.0 * len(cache[i] & cache[j])) / len(cache[i]) / len(cache[j])
    return table

def make_w2v_lookup_table(pairs_to_lookup):
    table = {}
    cache = VectorCache()
    w = WordCache()
    for (i, j) in pairs_to_lookup:
        if np.all(cache[i] < 1e-10) or np.all(cache[j] < 1e-10):
            table[i, j] = 0
            continue
        table[(i, j)] = np.dot(cache[i], cache[j]) / np.linalg.norm(cache[i]) / np.linalg.norm(cache[j])
        #if abs(table[i, j]) > 0.99 and len(w[i] & w[j]) == 0:
        #    table[i, j] = 0
        #table[(i, j)] = np.linalg.norm(cache[i] - cache[j])
    return table

def make_comp_diff_lookup_table(pairs_to_lookup):
    table = {}
    cache = VectorCache()
    for (i, j) in pairs_to_lookup:
        table[(i, j)] = np.abs(cache[i] - cache[j])
    return table

def make_ch_cosine_lookup_table(pairs_to_lookup):
    table = {}
    cache = CharCtrCache()
    for (i, j) in pairs_to_lookup:
        table[(i, j)] = (1.0 * sum((cache[i] & cache[j]).values())) / sum(cache[i].values()) / sum(cache[j].values())
    return table

def cosine_baseline():
    subdirs = get_subdirectories(all_data)

    # True data
    train_pos_pairs = set()
    train_neg_pairs = set()
    test_pos_pairs = set()
    test_neg_pairs = set()
    missing = set()

    for subdir in subdirs:
        pp, _np = get_pairs(subdirs[subdir])
        subdir_num = int(subdir)
        if subdir_num < 20:
            train_pos_pairs |= set(pp)
            train_neg_pairs |= set(_np)
        elif 20 < subdir_num <= 30:
            test_pos_pairs |= set(pp)
            test_neg_pairs |= set(_np)
            for cluster in subdirs[subdir].values():
                for item in cluster:
                    missing.add(item)

    cosine_table = make_cosine_lookup_table(reduce(lambda a, b: a | b,
        [train_pos_pairs, train_neg_pairs, test_neg_pairs,
            test_pos_pairs]))
    ch_cosine_table = make_ch_cosine_lookup_table(reduce(lambda a, b: a | b,
        [train_pos_pairs, train_neg_pairs, test_neg_pairs,
            test_pos_pairs]))

    # Generate train data
    a1 = np.array([[cosine_table[(i, j)], ch_cosine_table[(i, j)]] for (i, j) in train_pos_pairs])
    a2 = np.array([[cosine_table[(i, j)], ch_cosine_table[(i, j)]] for (i, j) in train_neg_pairs])
    poly = PolynomialFeatures(degree=2)
    Xtrain = poly.fit_transform(np.concatenate([a1, a2]))
    ytrain = np.concatenate([np.ones(len(train_pos_pairs)),
                        np.zeros(len(train_neg_pairs))])

    # Train Linear SVM
    model = SGDClassifier(loss='hinge', n_iter=10, n_jobs=-1)
    model.fit(Xtrain, ytrain)


    Xtest = poly.fit_transform(np.concatenate([
        np.array([[cosine_table[(i, j)], ch_cosine_table[i, j]] for (i, j) in test_pos_pairs]),
        np.array([[cosine_table[(i, j)], ch_cosine_table[i, j]] for (i, j) in test_neg_pairs])
    ]))
    ytest = np.concatenate([np.ones(len(test_pos_pairs)),
                        np.zeros(len(test_neg_pairs))])

    pred_pairs = model.predict(Xtest)

    all_test_pairs = list(test_pos_pairs) + list(test_neg_pairs)

    clusters = []
    for r, pair in zip(pred_pairs, all_test_pairs):
        if r == 1:
            clusters.append(list(pair))
            if pair[0] in missing: missing.remove(pair[0])
            if pair[1] in missing: missing.remove(pair[1])
    for it in missing:
        clusters.append([it])

    clusters = dict(enumerate(graph_agglomerate(clusters)))
    pred_pos, _ = get_pairs(clusters)

    compute_fscore(test_pos_pairs, set(pred_pos))

def word2vec_baseline():
    subdirs = get_subdirectories(all_data)

    # True data
    train_pos_pairs = set()
    train_neg_pairs = set()
    test_pos_pairs = set()
    test_neg_pairs = set()
    missing = set()
    actual = 0

    for subdir in subdirs:
        pp, _np = get_pairs(subdirs[subdir])
        subdir_num = int(subdir)
        if subdir_num < 20:
            actual += len(subdirs[subdir])
            train_pos_pairs |= set(pp)
            train_neg_pairs |= set(_np)
        elif 20 < subdir_num <= 30:
            test_pos_pairs |= set(pp)
            test_neg_pairs |= set(_np)
            for cluster in subdirs[subdir].values():
                for item in cluster:
                    missing.add(item)

    preprocess_vectors()
    w2v_cosine_table = make_w2v_lookup_table(reduce(lambda a, b: a | b,
        [train_pos_pairs, train_neg_pairs, test_neg_pairs,
            test_pos_pairs]))
    cosine_table = make_cosine_lookup_table(reduce(lambda a, b: a | b,
        [train_pos_pairs, train_neg_pairs, test_neg_pairs,
            test_pos_pairs]))
    ch_cosine_table = make_ch_cosine_lookup_table(reduce(lambda a, b: a | b,
        [train_pos_pairs, train_neg_pairs, test_neg_pairs,
            test_pos_pairs]))

    # Generate train data
    a1 = np.array([[w2v_cosine_table[i, j], cosine_table[i, j]] for (i, j) in train_pos_pairs])
    a2 = np.array([[w2v_cosine_table[i, j], cosine_table[i, j]] for (i, j) in train_neg_pairs])
    poly = PolynomialFeatures(degree=3)
    Xtrain = poly.fit_transform(np.concatenate([a1, a2]))
    ytrain = np.concatenate([np.ones(len(train_pos_pairs)),
                        np.zeros(len(train_neg_pairs))])

    # Train Logistic Regression
    model = LogisticRegression(n_jobs=-1)
    model.fit(Xtrain, ytrain)


    Xtest = poly.fit_transform(np.concatenate([
        np.array([[w2v_cosine_table[i, j], cosine_table[i, j]] for (i, j) in test_pos_pairs]),
        np.array([[w2v_cosine_table[i, j], cosine_table[i, j]] for (i, j) in test_neg_pairs])
    ]))
    ytest = np.concatenate([np.ones(len(test_pos_pairs)),
                        np.zeros(len(test_neg_pairs))])
    pred_pairs = model.predict(Xtest)
    print np.sum(pred_pairs)

    all_test_pairs = list(test_pos_pairs) + list(test_neg_pairs)

    clusters = []
    for r, pair in zip(pred_pairs, all_test_pairs):
        if r == 1:
            clusters.append(list(pair))
            if pair[0] in missing: missing.remove(pair[0])
            if pair[1] in missing: missing.remove(pair[1])
    for it in missing:
        clusters.append([it])

    clusters = dict(enumerate(graph_agglomerate(clusters)))
    print "Predicted", len(clusters)
    print Counter(map(len, clusters.values()))
    print "Actual", actual
    pred_pos, _ = get_pairs(clusters)

    compute_fscore(test_pos_pairs, set(pred_pos))
    write_gold_pairs(set(pred_pos))

def write_gold_pairs(pairs):
    normalize_pairs(pairs)
    outfile = open('predicted_gold_pairs.csv', 'w')
    df = all_data
    for (x, y) in pairs:
        outfile.write("%s,%s,%s,%s\n" %(df[2][x], df[3][x], 
            df[2][y], df[3][y]))
    outfile.close()



if __name__ == "__main__":
    word2vec_baseline()

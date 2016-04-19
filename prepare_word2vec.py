import pandas
import csv
import cPickle
from gensim.models import word2vec


df = pandas.read_csv('goldTruth.txt', sep=';', header=None,
		quoting=csv.QUOTE_NONE)

model = word2vec.Word2Vec.load_word2vec_format('/Users/vivek/Downloads/freebase-vectors-skipgram1000-en.bin', binary=True)

for i in xrange(len(df)):
	words = df[4][i].split()
	lemmas = df[5][i].split()
	result = []
	result_lemma = []
	flag = False
	for j in xrange(len(words) - 1):
		if flag: 
			flag = False
			continue
		bigram = words[j].lower() + '_' +  words[j+1].lower()
		label = '/en/' + bigram
		if label in model.vocab:
			flag = True
			result.append(bigram)
			result_lemma.append(lemmas[j] + '_' + lemmas[j+1])
		else:
			flag = False
			result.append(words[j])
			result_lemma.append(lemmas[j])
	if not flag:
		result.append(words[-1])
		result_lemma.append(words[-1])
	df.set_value(i, 4, ' '.join(result))
	df.set_value(i, 5, ' '.join(result_lemma))

vocab = reduce(lambda a, b: a | b, map(lambda s: set(s.split()), df[4]))
lemma = reduce(lambda a, b: a | b, map(lambda s: set(s.split()), df[5]))

vectors = {}

for word in (vocab | lemma):
    w1 = '/en/' + word
    w2 = '/en/' + word.lower()
    if w1 in model:
        vectors[word] = model[w1]
    elif w2 in model:
        vectors[word] = model[w2]

print len(vectors)
cPickle.dump(vectors, open('vectors_phrases.pkl', 'w'))

df.to_csv(open('goldTruthPhrases.txt', 'w'), sep=';', header=None, quoting=csv.QUOTE_NONE, index=False)

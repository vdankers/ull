from gensim.models.word2vec import Word2Vec
import logging
import pickle

logging.basicConfig(level="INFO")

# Prepare corpus
sentences = []
with open("data/europarl/training.en", encoding='utf8') as f:
    for line in f:
        if len(line.split()) <= 50:
            s = ["<s>"] + line.split() + ["</s>"]
            sentences.append(s)

# Learn model
model = Word2Vec(sentences, size=100, window=5, min_count=0, workers=4, sg=1, negative=5)

# Save vectors with pickle
vectors = dict()
for i, word in enumerate(model.wv.vocab):
    vectors[word] = model.wv[word]
pickle.dump(vectors, open("sgns.pickle", 'wb'))
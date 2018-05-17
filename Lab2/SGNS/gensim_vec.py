import logging
import gensim
from nltk.corpus import stopwords

sw = set(stopwords.words('english'))

#logging.basicConfig(level=logging.INFO)
sentences = open("../data/hansards/training.en").readlines()
sentences = [ [ w for w in s.split()  ] for s in sentences ]
model = gensim.models.word2vec.Word2Vec(sentences, size=50, window=5, min_count=1, workers=4, negative=15, sg=1, iter=10)
model.wv.save_word2vec_format("sgns_gensim_hansards.txt", binary=False)
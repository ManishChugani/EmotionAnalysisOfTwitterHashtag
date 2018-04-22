from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.cluster import KMeans
from gensim.models import FastText
import GetTweets as GT
sentences = GT.get_tweets()
#filename = 'GoogleNews-vectors-negative300.bin'
#model = KeyedVectors.load_word2vec_format(filename, binary=True)
model = FastText(sentences)
print(model)
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
#print(model.similarity('joy', 'man'))
print(result)

from gensim.models import Word2Vec
import gensim.downloader

# you can download a pretrained Word2Vec
# - or you can train your own model

# download a model
# 'glove-wiki-gigaword-300' (376.1 MB)
# 'word2vec-ruscorpora-300' (198.8 MB)
# '' (1.6 GB)
vectorizer = gensim.downloader.load('word2vec-google-news-300')

# train your own model
vectorizer = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4).wv
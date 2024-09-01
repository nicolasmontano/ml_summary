from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load or train a Word2Vec model
# For example, you can load Google's pre-trained model like this:
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Or train your own model:
sentences = [["this", "is", "a", "sample", "sentence"], ["another", "example", "sentence"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Step 2: Extract word vectors for words you want to visualize
words = list(model.wv.index_to_key)  # List of words in the vocabulary
word_vectors = np.array([model.wv[word] for word in words])

# Step 3: Apply t-SNE to reduce dimensions
tsne = TSNE(n_components=2, random_state=0, perplexity=6)
word_vectors_2d = tsne.fit_transform(word_vectors)

# Step 4: Plot the t-SNE result
plt.figure(figsize=(12, 8))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])

# Add labels to each point
for i, word in enumerate(words):
    plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.show()

import argparse
import numpy as np
from heapq import nsmallest
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

def read_vectors(file):
    vectors = {}
    file = open(file, 'r')
    vocab_size, vec_dim = file.readline().strip().split()
    print("Vocab size =", vocab_size, " Vector dimension =", vec_dim)
    for line in file.read().split('\n'):
        s = line.split()
        if not s: continue
        token, vector = s[0], np.array(s[1:], dtype=float)
        vectors[token] = vector
    file.close()
    return vectors, int(vocab_size), int(vec_dim)

def closest_words(word, vectors):
    similarity = {}
    if word not in vectors: return "That word does not exist in the vocabulary."
    for w in vectors: similarity[w] = cosine(vectors[word], vectors[w])
    return [word for word in nsmallest(5, similarity, key=similarity.get)]

def most_similar_words(a, b, c, vectors):
    similarity = {}
    if a not in vectors or b not in vectors or c not in vectors: return "One or more words do not exist in the vocabulary."
    vec = vectors[a] - vectors[b] + vectors[c]
    for w in vectors: similarity[w] = cosine(vec, vectors[w])
    return [word for word in nsmallest(5, similarity, key=similarity.get)]

def get_vectors(vec, vec_dim, words):
    vectors = np.zeros((len(words), vec_dim))
    for i, word in enumerate(words): vectors[i] = vec[word]
    return vectors

def tsne(med_vec, eng_vec_dim, words):
    vectors = get_vectors(med_vec, med_vec_dim, words)
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(vectors)
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mv", "--med_vec", type=str, help="Medical vectors file")
    parser.add_argument("-ev", "--eng_vec", type=str, help="English vectors file")
    args = parser.parse_args()

    print("Loading medical domain vectors")
    med_vec, med_vocab_size, med_vec_dim = read_vectors(args.med_vec)
    print("Loading plain english vectors")
    eng_vec, eng_vocab_size, eng_vec_dim = read_vectors(args.eng_vec)

    print("\n1. Closest words -> single word\n2. Most similar words -> three words (a - b + c)\n3. TSNE visualisation\n9. Exit\n")
    while True:
        query = input("Query Type > ")
        if query == "1":
            word = input("Word > ")
            print("From medical domain")
            print(closest_words(word, med_vec))
            print("From english domain")
            print(closest_words(word, eng_vec))
        elif query == "2":
            a, b, c = input("Words > ").split()
            print("From medical domain")
            print(most_similar_words(a, b, c, med_vec))
            print("From english domain")
            print(most_similar_words(a, b, c, eng_vec))
        elif query == "3":
            words = input("Words > ").split()
            tsne(med_vec, med_vec_dim, words)
            tsne(eng_vec, eng_vec_dim, words)
        else:
            quit()

import pickle
import argparse
import numpy as np
from heapq import nsmallest
from scipy.spatial.distance import cosine

def closest_words(word, vectors, vocab):
    similarity = {}
    if word not in vocab: return "That word does not exist in the vocabulary."
    for w in vocab: similarity[w] = cosine(vectors[vocab[word]], vectors[vocab[w]])
    return [word for word in nsmallest(5, similarity, key=similarity.get)]

def most_similar_words(a, b, c, vectors, vocab):
    similarity = {}
    if a not in vocab or b not in vocab or c not in vocab: return "One or more words do not exist in the vocabulary."
    vec = vectors[vocab[a]] + vectors[vocab[b]] - vectors[vocab[c]]
    for w in vocab: similarity[w] = cosine(vec, vectors[vocab[w]])
    return [word for word in nsmallest(5, similarity, key=similarity.get)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mv", "--med_vec", type=str, help="Medical vectors file", default="medical_vectors.npy")
    parser.add_argument("-mw", "--med_w2i", type=str, help="Medical word2idx file", default="medical_word2idx.pkl")
    parser.add_argument("-ev", "--eng_vec", type=str, help="English vectors file", default="english_vectors.npy")
    parser.add_argument("-ew", "--eng_w2i", type=str, help="English word2idx file", default="english_word2idx.pkl")
    args = parser.parse_args()

    med_vec = np.load(args.med_vec)
    med_w2i = pickle.load(open(args.med_w2i, "rb"))
    eng_vec = np.load(args.eng_vec)
    eng_w2i = pickle.load(open(args.eng_w2i, "rb"))

    print("1. Closest words -> single word\n2. Most similar words -> three words (a + b - c)\n3. Exit\n")
    while True:
        query = input("Query Type > ")
        if query == "1":
            word = input("Word > ")
            print("From medical domain")
            print(closest_words(word, med_vec, med_w2i))
            print("From english domain")
            print(closest_words(word, eng_vec, eng_w2i))
        elif query == "2":
            words = input("Words > ")
            a, b, c = words.split()
            print("From medical domain")
            print(most_similar_words(a, b, c, med_vec, med_w2i))
            print("From english domain")
            print(most_similar_words(a, b, c, eng_vec, eng_w2i))
        else:
            quit()

import argparse
from heapq import nsmallest
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

from SkipGram import SkipGram

def build_vocab(dataset, max_vocab_size):
    words = []
    vocab = {}
    word2index = {}

    for line in open(dataset, "r").readlines():
        for word in line.split():
            words.append(word)
            if word not in vocab: vocab[word] = 1
            else: vocab[word] += 1

    index = 0
    for word in sorted(vocab, key=vocab.get, reverse=True):
        if index >= max_vocab_size: break
        if word not in stopwords and not word.isdigit():
            word2index[word] = index
            index += 1

    print("Total tokens = %d" % len(words))
    print("Overall vocabulary = %d" % len(vocab))
    print("Model vocabulary = %d" % len(word2index))

    return words, word2index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ni", "--num_iters", type=int, help="Number of iterations", default=5)
    parser.add_argument("-hs", "--hidden_size", type=int, help="Hdden state size", default=30)
    parser.add_argument("-mv", "--max_vocab_size", type=int, help="Maximum size of vocabulary", default=20000)
    parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate of optimiser", default=0.01)
    parser.add_argument("-bl", "--bag_length", type=int, help="Size of context on each side", default=2)
    parser.add_argument("-ds", "--dataset", type=str, help="Dataset file")
    args = parser.parse_args()

    stopwords = set(stopwords.words('english'))
    words, vocab = build_vocab(args.dataset, args.max_vocab_size)
    model = SkipGram(vocab, args.hidden_size, args.learning_rate, args.bag_length)
    vectors = model.run_iterations(words, args.num_iters)

    while True:
        word = input("Word > ").strip()
        if word not in vocab:
            print("That word does not exist in the vocabulary. Try another one")
            continue

        similarity = {}
        for w in vocab: similarity[w] = cosine(vectors[vocab[word]], vectors[vocab[w]])
        most_similar = [word for word in nsmallest(10, similarity, key=similarity.get)]
        print(most_similar)

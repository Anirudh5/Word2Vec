import numpy as np

class SkipGram:
    def __init__(self, vocab, hidden_size=25, learning_rate=0.02, bag_length=2):
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.bag_length = bag_length
        self.vocab_size = len(vocab)
        self.matrix1 = self.init_matrix(self.vocab_size, hidden_size)
        self.matrix2 = self.init_matrix(hidden_size, self.vocab_size)

    def init_matrix(self, x, y):
        return np.zeros((x, y))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)

    def compute_loss(self, y, output):
        loss = 0.0
        for i in y:
            if output[i] <= 1e-5: return 1000000
            loss += -np.log(output[i])
        return loss / len(y)

    def backprop(self, y, word_ind, input, output):
        output *= len(word_ind)
        for i in word_ind: output[i] -= 1.0
        self.matrix1 -= self.learning_rate * np.outer(input, np.matmul(self.matrix2, output))
        self.matrix2 -= self.learning_rate * np.outer(np.matmul(self.matrix1.transpose(), input), output.transpose())
        return self.compute_loss(word_ind, output)

    def train(self, word_ind, y):
        input = np.zeros(self.vocab_size)
        input[y] = 1.0
        output = self.sigmoid(np.matmul(self.matrix1.transpose(), input))
        output = self.softmax(np.matmul(self.matrix2.transpose(), output))
        return self.backprop(y, word_ind, input, output)

    def run_iterations(self, words, num_iters):
        for epoch in range(num_iters):
            loss = 0
            index = -1
            while index + 2 * self.bag_length + 1 < len(words):
                index += 1
                if words[index + self.bag_length] not in self.vocab: continue
                word_ind = [self.vocab[words[i]] for i in range(index, index + self.bag_length) if words[i] in self.vocab]
                word_ind += [self.vocab[words[i]] for i in range(index + self.bag_length + 1, index + 2 * self.bag_length + 1) if words[i] in self.vocab]
                if len(word_ind) == 0: continue
                y = self.vocab[words[index + self.bag_length]]
                loss += self.train(word_ind, y)
            print("--- Epoch %d --- Total loss = %.2f ---" % (epoch + 1, loss))
        return self.matrix1

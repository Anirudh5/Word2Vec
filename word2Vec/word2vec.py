import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array

from Vocab import Vocab
from VocabItem import VocabItem
from UnigramTable import UnigramTable
from save import save

def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))

def init_net(dim, vocab_size):
    # Init syn0 with random numbers from a uniform distribution on the interval [-0.5, 0.5]/dim
    tmp = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    # Init syn1 with zeros
    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)

    return (syn0, syn1)

def train_process(pid):
    # Set fi to point to the right chunk of training file
    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)
    fi.seek(start)
    #print 'Worker %d beginning training at %d, ending at %d' % (pid, start, end)

    alpha = starting_alpha

    word_count = 0
    last_word_count = 0

    while fi.tell() < end:
        line = fi.readline().strip()
        # Skip blank lines
        if not line:
            continue

        # Init sent, a list of indices of words in line
        sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

        for sent_pos, token in enumerate(sent):
            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count

                # Recalculate alpha
                alpha = starting_alpha * (1 - float(global_word_count.value) / vocab.word_count)
                if alpha < starting_alpha * 0.0001: alpha = starting_alpha * 0.0001

                # Print progress info
                sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                                 (alpha, global_word_count.value, vocab.word_count,
                                  float(global_word_count.value) / vocab.word_count * 100))
                sys.stdout.flush()

            # Randomize window size, where win is the max window size
            current_win = np.random.randint(low=1, high=win+1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos+1:context_end] # Turn into an iterator?

            # CBOW
            if cbow:
                # Compute neu1
                neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                assert len(neu1) == dim, 'neu1 and dim do not agree'

                # Init neu1e with zeros
                neu1e = np.zeros(dim)

                # Compute neu1e and update syn1
                if neg > 0:
                    classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                else:
                    classifiers = zip(vocab[token].path, vocab[token].code)
                for target, label in classifiers:
                    z = np.dot(neu1, syn1[target])
                    p = sigmoid(z)
                    g = alpha * (label - p)
                    neu1e += g * syn1[target] # Error to backpropagate to syn0
                    syn1[target] += g * neu1  # Update syn1

                # Update syn0
                for context_word in context:
                    syn0[context_word] += neu1e

            # Skip-gram
            else:
                for context_word in context:
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        z = np.dot(syn0[context_word], syn1[target])
                        p = sigmoid(z)
                        g = alpha * (label - p)
                        neu1e += g * syn1[target]              # Error to backpropagate to syn0
                        syn1[target] += g * syn0[context_word] # Update syn1

                    # Update syn0
                    syn0[context_word] += neu1e

            word_count += 1

    # Print progress info
    global_word_count.value += (word_count - last_word_count)
    sys.stdout.write("\rAlpha: %f Progress: %d of %d (%.2f%%)" %
                     (alpha, global_word_count.value, vocab.word_count,
                      float(global_word_count.value)/vocab.word_count * 100))
    sys.stdout.flush()
    fi.close()

def __init_process(*args):
    global vocab, syn0, syn1, table, cbow, neg, dim, starting_alpha
    global win, num_processes, global_word_count, fi
    
    vocab, syn0_tmp, syn1_tmp, table, cbow, neg, dim, starting_alpha, win, num_processes, global_word_count = args[:-1]
    fi = open(args[-1], 'r')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        syn0 = np.ctypeslib.as_array(syn0_tmp)
        syn1 = np.ctypeslib.as_array(syn1_tmp)
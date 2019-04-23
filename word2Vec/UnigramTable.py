import argparse
import math
import struct
import sys
import time
import warnings

import numpy as np

from multiprocessing import Pool, Value, Array

class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """
    def __init__(self, vocab):
        vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab]) # Normalizing constant

        table_size = 100000000 # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print 'Filling unigram table'
        p = 0 # Cumulative probability
        i = 0
        for j, unigram in enumerate(vocab):
            p += float(math.pow(unigram.count, power))/norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]
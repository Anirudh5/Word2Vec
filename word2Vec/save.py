def save(vocab, syn0, fo, binary):
    print('Saving model to', fo)
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()

import re
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

def process_article(data):
    data = re.sub(r"\'s", "", data)
    data = sent_tokenize(data)
    for i, sent in enumerate(data):
        s = re.sub(r"[^a-z]+", " ", sent).split()
        s = [w for w in s if w not in stop and not len(w) < 3]
        data[i] = " ".join(s)
    data = "\n".join(data)
    return data + "\n"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    args = parser.parse_args()

    stop = set(stopwords.words('english'))
    output = open(args.output, "w")
    data = open(args.input, "r").read().lower().split('\n')
    for article in data:
        if article:
            output.write(process_article(article))
    output.close()

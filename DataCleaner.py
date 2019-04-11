import re
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    args = parser.parse_args()

    data = open(args.input, "r").read()
    data = data.lower()
    data = re.sub(r"\W+", " ", data)
    data = " ".join(data.split())
    output = open(args.output, "a")
    output.write(data)
    output.close()

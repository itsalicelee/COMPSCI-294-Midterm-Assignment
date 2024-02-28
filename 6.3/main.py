import random 
import string
from collections import Counter
import heapq
import argparse

def genRandomString(n):
    '''Generates a random string of lowercase letters of length n'''
    return ''.join(random.choices(string.ascii_lowercase, k=n))

class Node:
    def __init__(self, left=None, right=None, freq=0, char=''):
        self.left = left
        self.right = right
        self.freq = freq
        self.char = char  # the original char
        self.val = ''  # the encoded value 

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanTree:
    def __init__(self, text, args):
        self.text = text
        self.freq = Counter(text)
        self.root = None
        self.encodedDict = {}
        self.args = args

    def encode(self):
        q = [Node(freq=freq, char=char) for char, freq in self.freq.items()]
        heapq.heapify(q)

        while len(q) > 1:
            left = heapq.heappop(q)
            right = heapq.heappop(q)
            # assign '0' to the left child and '1' to the right child
            left.val, right.val = '0', '1'
            node = Node(left=left, right=right, freq=left.freq + right.freq)
            heapq.heappush(q, node)
        self.root = q[0]

    def traverse(self, root, s):
        if not root:
            return

        if not root.left and not root.right:
            self.encodedDict[root.char] = s

        self.traverse(root.left, s + '0')
        self.traverse(root.right, s + '1')

    def printCompressRatioInfo(self):
        # Encode the text
        self.encode()
        self.traverse(self.root, '')
        orig_bits = len(self.text) * 8  # 8 bits per character
        if self.args.show_orig:
            print('===== Text and Character Frequency =====')
            print('Text: {}\nNumber of bits: {}'.format(self.text, orig_bits))

            for k, v in self.freq.items():
                print(k + ": " + str(v))
        if self.args.show_encoded:
            print('===== Encoded Dictionary =====')
            for k, v in self.encodedDict.items():
                print(k + ": " + v)

        print('===== Compression Ratio =====')
        print('Original number of bits: ', orig_bits)
        # Calculate the number of bits after compression
        compressed_bits = sum(len(self.encodedDict[char]) * self.freq[char] for char in self.freq)
        print('Compressed number of bits: ', compressed_bits)

        # Calculate the compression ratio
        compression_ratio = (len(self.text) * 8) / compressed_bits
        print('Compression Ratio: {:.2f}'.format(compression_ratio))

def arg_parse():
    parser = argparse.ArgumentParser(description='Exercise 6.3 Huffman Compression')
    parser.add_argument('--n', default=10, type=int, help="random string length (default: 10)")
    parser.add_argument('--show-orig', default=True, type=bool, help="show original text and bits (default: True)")
    parser.add_argument('--show-encoded', default=True, type=bool, help="show encoded characters (default: True)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    huffman = HuffmanTree(genRandomString(args.n), args)
    huffman.printCompressRatioInfo()

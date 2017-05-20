import argparse
import pickle as pkl
from get_data import CBOWCorpusReader
from cbow import CBOW
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='tinyshakespear', help='Corpus file path')
parser.add_argument('--ngram_size', type=int, default=3, help='Number of grams to be considered')
parser.add_argument('--vocab_size', type=int, help='Number of unique words')
parser.add_argument('--embedding_dim', type=int, default=200, help='Embedding layer size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Model learning rate value')
parser.add_argument('--activation_function', type=str, default="tanh", help='Hidden layer activation function')
parser.add_argument('--optimizer', type=str, default='SGD', help='Model optimization algorithm')
parser.add_argument('--loss_function', type=str, default='ce', help='Loss function that would be minimized')
parser.add_argument('--epochs', type=int, default=10, help='Number of iterations')

args = parser.parse_args()

print("Begin Reading Corpus Data and Tokenizing")
data_reader = CBOWCorpusReader(args.corpus)
grams = data_reader.get_ngram_words()
words_freq = data_reader.get_words_frequency()
word2idx = data_reader.get_word2idx()
idx2word = data_reader.get_idx2word()
print("End Reading the Data")

args.vocab_size = len(words_freq)
cbow = CBOW(args)
cbow.init_session()
cbow.build()

print("Begin Training")
learning_curve = []

for epoch in range(0, args.epochs):
    error = 0.0
    print(epoch)
    for batch in grams:
        x_input, y_output, x_input_reshape = [], [], []

        for item in batch:
            def get_one_hot(idx):
                one_hot = ([0] * (args.vocab_size + 1))
                one_hot[idx] = 1

                return one_hot

            x_input.append(list(itertools.chain.from_iterable(list(map(lambda v: get_one_hot(word2idx[v]), item[0])))))
            y_output.append(get_one_hot(word2idx[item[1][0]]))

        error += cbow.run(x_input, y_output)

    learning_curve.append(error)


___author__ = 'Ahmed Hani Ibrahim'


class CBOWCorpusReader(object):
    def __init__(self, file_name):
        self.__file_name = file_name

        self.__read()

    def __read(self):
        with open(self.__file_name, 'r') as reader:
            self.__all_sentences = reader.readlines()
            self.__words_frequency = {}
            self.__word2idx = {}
            self.__idx2word = {}
            idx = 0

            for sentence in self.__all_sentences:
                sentence_tokens = sentence.strip().split()

                if len(sentence_tokens) == 0:
                    continue

                for token in sentence_tokens:
                    if token in self.__words_frequency:
                        self.__words_frequency[token] += 1
                    else:
                        self.__words_frequency[token] = 1
                        self.__word2idx[token] = idx
                        self.__idx2word[idx] = token
                        idx += 1
            self.__word2idx['<end>'] = idx
            self.__idx2word[idx] = '<end>'

    def get_ngram_words(self, ngram_size=3, batch_size=2):
        ngram_size -= 1
        grams = []
        batched_grams = []

        for sentence in self.__all_sentences:
            sentence_tokens = sentence.strip().split()
            batch_counter = 0

            for i in range(0, len(sentence_tokens)):
                input_ = []

                for j in range(0, ngram_size):
                    word = "<end>" if (i + j) >= len(sentence_tokens) else sentence_tokens[i + j]
                    input_.append(word)

                word = "<end>" if i + ngram_size >= len(sentence_tokens) else sentence_tokens[i + ngram_size]
                grams.append((input_, [word]))
                batch_counter += 1

                if batch_counter == batch_size:
                    batched_grams.append(grams)
                    grams = []
                    batch_counter = 0

        return batched_grams

    def get_words_frequency(self):
        return self.__words_frequency

    def get_word_frequency(self, word):
        return self.__words_frequency[word] if word in self.__words_frequency else None

    def get_word2idx(self):
        return self.__word2idx

    def get_word_idx(self, word):
        return self.__word2idx[word] if word in self.__word2idx else None

    def get_idx2word(self):
        return self.__idx2word

    def get_idx_word(self, idx):
        return self.__idx2word[idx] if idx in self.__idx2word else None

dr = CBOWCorpusReader("tinyshakespear")
grams = dr.get_ngram_words()
words_freq = dr.get_words_frequency()
word2idx = dr.get_word2idx()
idx2word = dr.get_idx2word()
MAX_LENGTH = 15


class Vocab:
    def __init__(self,name):
        self.name = name
        self.word2id = {}
        self.word2count = {}
        self.id2word = {0: 'PAD', 1: 'UNK', 2: 'SOS', 3: 'EOS'}
        self.n_words = 4

    def add_sentence(self,sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self,word):
        if word not in self.word2id:
            self.word2id[word] = self.n_words  # self.n_words 是词典中词的总数
            self.word2count[word] = 1  # word2count 是词频
            self.id2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __len__(self):
        return len(self.id2word)

    def get(self,item):
        return self.word2id[item]


def read_data(first_txt, second_txt):

    first_lines = open(first_txt, encoding='utf-8').read().strip().split('\n')
    second_lines = open(second_txt, encoding='utf-8').read().strip().split('\n')

    pairs = [[f, s] for f, s in zip(first_lines, second_lines)]

    input_vocab = Vocab("上联")
    output_vocab = Vocab("下联")

    return input_vocab, output_vocab, pairs


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def add_vocab(first_path,second_path):
    input_lang, output_lang, pairs = read_data(first_path, second_path)
    pairs = filter_pairs(pairs)
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs
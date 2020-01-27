import sys
from enum import Enum

import fasttext as fastText
import numpy as np

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class FileFormat(Enum):
    CONLL = 1
    TOKEN = 2
    TEXT = 3


class FileStream(Enum):
    FILE = 1
    SYSTEM = 2
    DIRECT = 3


class Embeddings(object):

    def getEmbeddingVector(self):
        return None

    def load(self, filename):
        return None

    def save(self, outputfile):
        return None


class Unknown(Enum):
    UNKNOWN = 1
    KNOWN = 2
    UNKNOWN_ADD = 3


class Word(object):
    def __init__(self, word, processed_word, identifier, unknown=Unknown.KNOWN):
        self.word = word
        self.processed_word = processed_word
        self.identifier = identifier
        self.unknown = unknown

    def __str__(self):
        return self.processed_word

    def __repr__(self):
        return self.identifier


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """

    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None, file_format=FileFormat.CONLL, stream=FileStream.FILE):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
        self.file_format = file_format
        self.stream = stream

    def __iter__(self):
        if self.stream == FileStream.SYSTEM:
            file_stream = sys.stdin
        elif self.stream == FileStream.FILE:
            file_stream = open(self.filename)
        else:
            from io import StringIO
            file_stream = StringIO(self.filename)

        words, tags = [], []
        niter = 0
        if self.file_format == FileFormat.CONLL:
            for line in file_stream:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split()
                    word, tag = ls[0], ls[-1]

                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]

            if len(words) > 0:
                yield words, tags
        elif self.file_format == FileFormat.TOKEN:
            for line in file_stream:
                niter += 1
                pre_words = line.strip().split()
                for w in pre_words:
                    words += [self.processing_word(w)]
                if self.max_iter is not None and niter > self.max_iter:
                    break
                tags = ["O"] * len(words)
                yield words, tags
                words, tags = [], []
            if len(words) > 0:
                yield words, tags
        elif self.file_format == FileFormat.TEXT:
            from nltk.tokenize import sent_tokenize, word_tokenize
            lines = ""
            for line in file_stream:
                lines += " " + line
            lines = lines.strip()
            for sentence in sent_tokenize(lines):
                niter += 1
                pre_words = word_tokenize(sentence)
                for w in pre_words:
                    words += [self.processing_word(w)]
                if self.max_iter is not None and niter > self.max_iter:
                    break
                tags = ["O"] * len(words)
                yield words, tags
                words, tags = [], []
            if len(words) > 0:
                yield words, tags
        else:
            sys.stderr.write("Format not supported")

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:

            for w in words:
                vocab_words.add(str(w))
            # vocab_words.update(word_str)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, encoding="utf8") as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(str(word)))
            else:
                f.write(str(word))
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        idx = 0
        with open(filename, encoding="utf8") as f:
            for word in f:
                word = word.strip()
                if word in d:
                    continue
                d[word] = idx
                idx += 1
            # for idx, word in enumerate(f):
            #    word = word.strip()
            #    d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def load_vocab_rev(filename):
    d = {}
    with open(filename, encoding="utf8") as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[idx] = word
    return d


def get_oov_embeddings(config):
    if config.oov_size == 0:
        return
    if not config.embedding_type == "fasttext":
        sys.stderr.write("OOV replacement only works with fasttext!\n")
        sys.exit(0)

    model = fastText.load_model(config.filename_embeddings)
    oov_embeddings = np.zeros((config.oov_size, config.dim_word))

    i = 0
    for word in config.oov_words:
        oov_embeddings[i,] = model.get_word_vector(word)
        config.vocab_words[word] = len(config.vocab_words)
        i += 1
    all_words = model.get_words()
    import random

    for i in range(len(config.oov_words), config.oov_size):
        rand_word_id = random.randint(0, len(all_words))
        rand_word = all_words[rand_word_id]
        oov_embeddings[i,] = model.get_word_vector(rand_word)
    return oov_embeddings


def export_trimmed_embedding_vectors(vocab, glove_filename, trimmed_filename, dim, embedding_type):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
        type: type of embeddings: fasttext, glove, w2v
    """
    embeddings = np.zeros([len(vocab), dim])
    if embedding_type.lower() == "glove":
        with open(glove_filename, encoding="utf-8") as f:
            print("Using glove..")
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)
    elif embedding_type.lower() == "w2v":
        i = 0
        for l in open(glove_filename):
            i += 1
            if i == 1:
                continue
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    elif embedding_type.lower() == "fasttext":
        print("Loading fasttext model ..")
        model = fastText.load_model(glove_filename)
        print("Model loaded.")
        for w in vocab:
            word_idx = vocab[w]
            embeddings[word_idx] = model.get_word_vector(w)
    else:
        raise Exception("Embedding type not known " + embedding_type)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_embedding_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        if not filename.endswith("npz"):
            # if not is_glove:
            #    return np.loadtxt(filename,skiprows=1)
            return np.loadtxt(filename)
        with np.load(filename) as data:
            print('Loaded embeddings')
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_tag(vocab):
    """
    Args:
        vocab: vocabulary of the tagset

    Returns:
        a function that returns the id for a specific tag

    """

    def f(tag):
        if tag in vocab:
            tag_id = vocab[tag]
            return tag_id;
        else:
            raise Exception("Unknow key is not allowed. Check that " \
                            "your vocab (tags?) is correct")

    return f


def preprocessing_word(word, lowercase):
    # 1. preprocess word
    processed_word = word
    if lowercase:
        processed_word = word.lower()
    if word.isdigit():
        processed_word = NUM
    return processed_word


def add_oov_words(datasets, config):
    oov_size = config.oov_size
    if oov_size == 0:
        return
    for dataset in datasets:
        for words, tags in dataset:
            for char, word in words:
                pword = word.processed_word
                if word.unknown == Unknown.UNKNOWN:
                    if pword in config.vocab_words:
                        word.unknown = Unknown.UNKNOWN_ADD
                        word.identifier = config.vocab_words[pword]
                        continue

                    if oov_size > config.oov_current_size:
                        config.vocab_words[pword] = len(config.vocab_words)
                        word.unknown = Unknown.UNKNOWN_ADD
                        word.identifier = config.vocab_words[pword]
                        config.oov_words.append(pword)
                        config.oov_current_size += 1


def get_processing_word(config):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """

    # set the parameters based on the configuration object
    vocab_words = None
    if hasattr(config, 'vocab_words'):
        vocab_words = config.vocab_words
    vocab_chars = None
    if hasattr(config, 'vocab_chars'):
        vocab_chars = config.vocab_chars
    lowercase = config.lowercase

    use_chars = config.use_chars
    allow_unk = True
    if hasattr(config, 'allow_unk'):
        allow_unk = config.allow_unk

    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and use_chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        processed_word = preprocessing_word(word, lowercase)

        # 2. get id of word
        unknown = Unknown.KNOWN
        word_id = -1


        if vocab_words is not None:
            if processed_word in vocab_words:
                word_id = vocab_words[processed_word]
                if processed_word in config.oov_words:
                    unknown = Unknown.UNKNOWN_ADD
            else:
                if allow_unk:
                    word_id = vocab_words[UNK]
                    unknown = Unknown.UNKNOWN
                else:
                    raise Exception("Unknow key is not allowed. Check that " \
                                    "your vocab (tags?) is correct")
        w = Word(word, processed_word, word_id, unknown)
        # 3. return tuple char ids, word id
        if vocab_chars is not None and use_chars == True:
            return char_ids, w
        else:
            return w

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        if type(x[0]) == tuple:
            x = list(zip(*x))
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]

    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1

        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

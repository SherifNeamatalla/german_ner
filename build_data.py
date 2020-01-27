import os
import sys
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_embedding_vectors, get_processing_word


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    
    if len(sys.argv)<2:
        sys.stderr.write("Too few arguments have been specified\n")
        sys.stderr.write("python "+sys.argv[0]+" config [additional vocabulary in conll format]\n")
        sys.exit(0)    
    # get config and processing of words
    config_file = sys.argv[1]
    
    config = Config(config_file,load=False)
    processing_word = get_processing_word(config)
#    processing_word = get_processing_word(lowercase=config.lowercase)

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)
    train = CoNLLDataset(config.filename_train, processing_word)
    

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    #add additional tags/vocabulary where the data is applied to!
    if len(sys.argv)>2:
        for i in range(2,len(sys.argv)):
            wo,tg = get_vocabs([CoNLLDataset(sys.argv[i],processing_word)])
            vocab_words |=  wo
            vocab_tags |=  tg
    #if config.use_pretrained:
    #    vocab_glove = get_vocab(config.filename_embeddings)
    #if config.use_pretrained:
    #    vocab = vocab_words & vocab_glove
    #else:
    vocab = vocab_words
    vocab.add(UNK)

    vocab.add(NUM)
    # Save vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)
    
    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)

    if config.use_pretrained:
        export_trimmed_embedding_vectors(vocab, config.filename_embeddings,
                                config.filename_embeddings_trimmed, config.dim_word, config.embedding_type)

    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)

   
        
if __name__ == "__main__":
    main()

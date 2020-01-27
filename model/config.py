import os,sys


from .general_utils import get_logger
from .data_utils import Word
from .data_utils import get_trimmed_embedding_vectors, load_vocab, \
        get_processing_word,get_processing_tag
import configparser

from enum import Enum

class EnvInterpolation(configparser.BasicInterpolation):
    """Interpolation which expands environment variables in values."""
    def before_get(self, parser, section, option, value, defaults):
        print("----------")
        print( value)
        print(option)
        print(defaults)
        return os.path.expandvars(value)

#        if value=="$PWD":
            
#            print(os.path.expandvars(value))
#            defaults[option]=os.path.expandvars(value)
#            return os.path.expandvars(value)
#        else:
#            L = []
#            self._interpolate_some(parser, option, L, value, section, defaults, 1)
#            return ''.join(L)
        #    return value
#        else:
            #newvalue = super(EnvInterpolation,self).before_get(parser,section,option,value,defaults)
            #print(newvalue)
            #return newvalue


    

class Config():
    def __init__(self, configfile, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        self.setParameters(configfile)
        # directory for training outputs
        if not os.path.exists(self.dir_vocab_output):
            os.makedirs(self.dir_vocab_output)
        if not os.path.exists(self.dir_model_output):
            os.makedirs(self.dir_model_output)
        # create instance of logger
        self.logger = get_logger(self.path_log)

        self.pad_token = Word("","", 0, False)
        
        # load if requested (default)
        if load:
            print('Loading trimmed embeddings..')
            print(self.filename_embeddings_trimmed)
            self.load()
        
    
    def setParameters(self,configfile):
        param = configparser.SafeConfigParser(interpolation = configparser.ExtendedInterpolation())
        #param = configparser.SafeConfigParser(os.environ)#interpolation = EnvInterpolation())
#        param = configparser.SafeConfigParser(interpolation = )
        configfile_str = open(configfile).read()
        #configfile_str = os.path.expandvars(configfile_str)
        configfile_str = configfile_str.replace("$PWD",os.path.abspath(os.path.dirname(configfile)))
        param.read_string(configfile_str)
        self.dir_model_output=param.get('PATH','dir_model_output')
        self.dir_vocab_output=param.get('PATH','dir_vocab_output')
        self.dir_model=param.get('PATH','dir_model')
        self.path_log=param.get('PATH','path_log')
        


        self.filename_train=param.get("PATH",'filename_train')
        self.filename_test=param.get('PATH','filename_test')
        self.filename_dev=param.get('PATH','filename_dev')
       
        self.filename_words = param.get('PATH','filename_words')
        self.filename_tags = param.get('PATH','filename_tags')
        self.filename_chars = param.get('PATH','filename_chars')

        #embedding types: Glove, w2v, fasttext
        try:
            self.embedding_type = param.get("EMBEDDINGS","embedding_type")
        except configparser.NoOptionError:
            self.embedding_type = "Glove"

        self.dim_word=param.getint('EMBEDDINGS','dim_word')
        self.dim_char=param.getint('EMBEDDINGS','dim_char')
        self.filename_embeddings=param.get('EMBEDDINGS','filename_embeddings')
        self.filename_embeddings_trimmed=param.get('EMBEDDINGS','filename_embeddings_trimmed')
        self.use_pretrained=param.getboolean('EMBEDDINGS','use_pretrained')
        
        self.use_large_embeddings = param.getboolean('EMBEDDINGS','use_large_embeddings')
        
        self.oov_size = 0
        self.oov_current_size = 0
        print('Trimmed Embeddings used : ',self.filename_embeddings_trimmed)
        print('Embeddings name : ',self.filename_embeddings)

        if param.has_option('EMBEDDINGS','oov_size'):
            self.oov_size = param.getint('EMBEDDINGS','oov_size')

        self.max_iter=param['PARAM']['max_iter']
        
        if not self.embedding_type == "fasttext" and self.oov_size>0:
            sys.stderr.write("Embeddings for unknown words cannot be generated for "+self.embedding_type+", thus the parameter oov_size will be set to zero\n")
            self.oov_size= 0
        
        if self.use_pretrained== False and self.use_large_embeddings == True:
            sys.stderr.write("If you want to train embeddings from scratch the use_large_embedding option is not valid\n")

        if (self.max_iter=="None") :
            self.max_iter = None
        self.train_embeddings=param.getboolean('PARAM','train_embeddings')
        
        try:
            self.lowercase = param.getboolean("PARAM","lowercase")
        except configparser.NoOptionError:
            self.lowercase = True

        
        self.nepochs=   param.getint('PARAM','nepochs')
        self.dropout=   param.getfloat('PARAM','dropout')
        self.batch_size=param.getint('PARAM','batch_size')
        self.lr_method=param.get('PARAM','lr_method')
        self.lr=param.getfloat('PARAM','lr')
        self.lr_decay=param.getfloat('PARAM','lr_decay')
        self.clip=param.getint('PARAM','clip')
        self.nepoch_no_imprv=param.getint('PARAM','nepoch_no_imprv')
        self.hidden_size_char=param.getint('PARAM','hidden_size_char')
        self.hidden_size_lstm=param.getint('PARAM','hidden_size_lstm')
        self.use_crf = param.getboolean('PARAM','use_crf')
        self.use_chars = param.getboolean('PARAM','use_chars')
        
        self.oov_words = []
        
    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self)
        
        self.processing_tag  = get_processing_tag(self.vocab_tags)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_embedding_vectors(self.filename_embeddings_trimmed)
                if self.use_pretrained else None)
        

    # general config
    

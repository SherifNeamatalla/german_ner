from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import sys

def main():
    # create instance of config
    config_file = sys.argv[1]
    config = Config(config_file)

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    model.train(train, dev)

if __name__ == "__main__":
    main()

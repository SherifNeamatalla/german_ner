import sys

from model.config import Config
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main():
    # create instance of config
    config_file = sys.argv[1]

    config = Config(config_file)

    print("dir model : ", config.dir_model)

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)
    # model.reinitialize_weights("words")
    # model.reinitialize_weights("chars")
    # model.reinitialize_weights("train_step")

    # Evaluate on another data set
    if len(sys.argv) > 2:
        test_file_name = sys.argv[2]
        test = CoNLLDataset(test_file_name, config.processing_word,
                            config.processing_tag, config.max_iter)
        print("Testing on ", test_file_name, "..")


    # create dataset
    else:
        test = CoNLLDataset(config.filename_test, config.processing_word,
                            config.processing_tag, config.max_iter)
        print("Testing on ", config.filename_test, "..")

        # evaluate and interact
    # model.predict_test(test, output=sys.stdout)
    model.evaluate(test)

    #interactive_shell(model)


if __name__ == "__main__":
    main()

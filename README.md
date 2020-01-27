# Named Entity Recognition with Tensorflow



This repository contains a NER implementation using Tensorflow (based on BiLSTM + CRF and character embeddings) that is based on the implementation by [Guillaume Genthial](https://github.com/guillaumegenthial/sequence_tagging). We have modified this implementation including its documentation. The major changes are listed below:


Mainly, we have done the following changes:
- convert from python 2 to python 3
- extract parameters from source code to a single config file
- create new script for testing new files
- create new script and modify source code for simple transfer learning
- support for several embeddings (GloVe, fasttext, word2vec)
- support to load all embeddings of a model
- support to dynamically load OOV embeddings during testing

Currently, we only provide models for contemporary German and historic German texts.


Table of Content
================


 - [Task of Named Entity Recognition](#task-of-named-entity-recognition)
 - [Machine Learning Model](#machine-learning-model)
 - [Requirements](#requirements)
 - [Run an Existing Model](#run-an-existing-model)
 - [Download Models and Embeddings](#download-models-and-embeddings)
   * [Manual Download](#manual-download)
   * [Automatic Download](#automatic-download)
 - [Train a New Model](#train-a-new-model)
 - [Transfer Learning](#transfer-learning)
 - [Predict Labels for New Text](#predict-labels-for-new-text)
 - [Server for Predicting Labels for New Text](#server-for-predicting-labels-for-new-text)
 - [Parameters in the Configuration File](#parameters-in-the-configuration-file)
 - [Requirements](#requirements) 
 - [Citation](#citation)
 - [License](#license)
  







## Task of Named Entity Recognition

The task of Named Entity Recognition (NER) is to predict the type of entity. Classical NER targets on the identification of locations (LOC), persons (PER), organization (ORG) and other (OTH). Here is an example

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```


## Machine Learning Model

The model is similar to [Lample et al.](https://arxiv.org/abs/1603.01360) and [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf). A more detailed description can be found [here](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe, Word2Vec, FastText here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF

## Requirements


To run the python code, you need python3 and the requirements from the following [file](https://github.com/riedlma/sequence_tagging/blob/master/requirements.txt) which can be easily installed:

```
pip3 install -r requirements.txt
```

In addition, you need to build fastText manually, as described [here](https://github.com/facebookresearch/fastText/tree/master/python), which are the following commands:

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip3 install .
```
Windows user might face [problems](https://github.com/salestock/fastText.py/issues/91) installing the fastText package. One of the solutions seems to be to install the "[Visual C++ 2015 Build Tools](https://www.microsoft.com/en-us/download/details.aspx?id=48159)".


## Run an Existing Model

To run pre-computed models, you need to install the [required python packages](#requirements) and you need to download the model and the embeddings. This can be done automatically with a python script as described [here](#automatic-download). However, the models and the embeddings can also be downloaded manually as described [here](#manual-download).


Here, we will fully describe, how to apply the best performing GermEval model to a new file.
First, we need to download the project, the model and the embeddings:

```
git clone https://github.com/riedlma/sequence_tagging
cd sequence_tagging
python3 download_model_embeddings.py GermEval
```

Now, you can create a new file (called test.conll) that should contain one token per line and might contain the following content:

```
Diese 
Beispiel
wurde
von
Martin
Riedl
in
Stuttgart 
erstellt
.
``` 

To start the entity tagging, you run the following command:

```
python3 test.py model_transfer_learning_conll2003_germeval_emb_wiki/config test.conll 
```

The output should be as following:

```
Diese diese	KNOWN	O
Beispiel beispiel	KNOWN	O
wurde wurde	KNOWN	O
von von	KNOWN	O
Martin martin	KNOWN	B-PER
Riedl riedl	KNOWN	I-PER
in in	KNOWN	O
Stuttgart stuttgart	KNOWN	B-LOC
erstellt erstellt	KNOWN	O
. .	KNOWN	O
```
The first column is the input word, the second column specifies the pre-processed word (here lowercased). The third column contains a flag, whether the word has been known during training (KNOWN) or not (UNKNOWN). If labels are assigned to the input file they will be in the third column. Otherwise, they will not be contained. The last column contains the predicted tags.



## Download Models and Embeddings
We provide the best performing model for the following datasets:

### Datasets

| Name| Language | Description| Webpage| 
|-----|----------|------------|---------|
| CoNLL 2003 | German | NER dataset based on Newspaper | [link](https://www.clips.uantwerpen.be/conll2003/ner/)
| GermEval 2014 | German | NER dataset based on Wikipedia | [link](https://sites.google.com/site/germeval2014ner/)|
| ONB| German |NER dataset based on texts of the Austrian National Library from 1710 and 1873 |[link](http://github.com/KBNLresearch/europeananp-ner/)|
| LFT | German | NER dataset based on text of the Dr Friedrich Teßmann Library from 1926 | [link](http://github.com/KBNLresearch/europeananp-ner/)|
| ICAB-NER09| Italian | NER dataset for Italian | [link](http://ontotext.fbk.eu/icab.html) |
| CONLL2002-NL | Dutch | NER for Dutch | [link](https://www.clips.uantwerpen.be/conll2002/) |

All provided models are trained using transfer learning techniques. The models and the embeddings can be downloaded [manually](#manual-download) or [automatically](#automatic-download).


### Manual Download of Models

The models can be downloaded as described in the table. The models should be stored directly on the project directory. Furthermore, they need to be uncompressed (tar xfvz \*tar.gz)

|  Optimized for | Trained | Transfer Learning |Embeddings| Download|
|----------------|------------|---------|-----|------|
| GermEval 2014 | CoNLL2003| GermEval 2014 | German Wikipedia|[link](http://www2.ims.uni-stuttgart.de/data/ner_de/models/model_transfer_learning_conll2003_germeval_emb_wiki.tar.gz) |
| CoNLL 2003 (German) | GermEval 2014 | CoNLL 2003 | German Wikipedia|[link](http://www2.ims.uni-stuttgart.de/data/ner_de/models/model_transfer_learning_conll2003_germeval_emb_wiki.tar.gz) |
| ONB | GermEval 2014 | ONB | German Europeana |  [link](http://www2.ims.uni-stuttgart.de/data/ner_de/models/model_transfer_learning_germeval_onb_emb_euro.tar.gz) |
| LFT | GermEval 2014 | LFT | German Wikipedia | [link](http://www2.ims.uni-stuttgart.de/data/ner_de/models/model_transfer_learning_germeval_lft_emb_wiki.tar.gz) |
|ICAB-NER09 | ICAB-NER09 | none | Italian Wikipedia | [link](http://www2.ims.uni-stuttgart.de/data/ner_de/models/model_ner_wiki_it.tar.gz) |
|CONLL2002-NL | CONLL2002-NL | none | Dutch newspaper | [link](http://www2.ims.uni-stuttgart.de/data/ner_de/models/model_ner_conll2002_nl.tar.gz) 

The embeddings should best be stored in the folder *embeddings* inside the project folder.
We provide the full embeddings (named Complete) and the filtered embeddings, which only contain the vocabulary of the data of the task. These filtered models have also been used to train the pre-computed models. The German Wikipedia model is provided by [Facebook Research](http://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).

| Name | Computed on | Dimensions | Complete  | Filtered|
|------|-------------|------------|-----------|---------|
| Wiki | German Wikipedia | 300   | [link](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.zip)  |  [link](http://www2.ims.uni-stuttgart.de/data/ner_de//embeddings/fasttext.wiki.de.bin.trimmed.npz)|
| Euro | German Europeana | 300   |  [link](http://www2.ims.uni-stuttgart.de/data/ner_de//embeddings/fasttext.german.europeana.skip.300.bin) | [link](http://www2.ims.uni-stuttgart.de/data/ner_de//embeddings/fasttext.german.europeana.skip.300.bin.trimmed.npz) |
| Wiki | Italian Wikipedia | 300   | [link](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.it.zip)  |  [link](http://www2.ims.uni-stuttgart.de/data/ner_de//embeddings/fasttext.wiki.it.bin.trimmed.npz)|
| Wiki | Dutch Wikipedia | 300   | [link](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.nl.zip)  |  [link](http://www2.ims.uni-stuttgart.de/data/ner_de//embeddings/fasttext.wiki.nl.bin.trimmed.npz)|

### Automatic Download of Models

Using the python script *download_model_embeddings.py* the models and the embeddings can be donwloaded automatically. In addition, the files are placed at the recommended location and are uncompressed.  You can choose between the several options:

```
~ user$ python3 download_model_embeddings.py 

No download option has been specified:
python download_model_embeddings.py options

Following download options are possible:
all                 download all models and embeddings
all_models          download all models
all_embed           download all embeddings
eval                download CoNLL 2003 evaluation script
GermEval            download best model and embeddings for GermEval
CONLL2003           download best model and embeddings for CONLL2003
ONB                 download best model and embeddings for ONB
LFT                 download best model and embeddings for LFT
ICAB-NER09-Italian  download best model and embeddings for ICAB-NER09-Italian
CONLL2002-NL        download best model and embeddings for ICAB-NER09-Italian


```


## Train a New Model

We will describe how a new model can be trained and describe it based on training a model on the GermEval 2014 dataset using pre-computed word embeddings from German Wikipedia. First, we need to download the training data. For training a model, we expect files to have two columns with the first column specifying the word and the second column containing the label. 

```
mkdir -p corpora/GermEval
wget -O corpora/GermEval/NER-de-train.tsv  https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv
wget -O corpora/GermEval/NER-de-dev.tsv  https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv
wget -O corpora/GermEval/NER-de-test.tsv  https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv
cat corpora/GermEval/NER-de-train.tsv  | grep -v "^[#]" | cut -f2,3 |  sed "s/[^ \t]\+\(deriv\|part\)$/O/g" > corpora/GermEval/NER-de-train.tsv.conv
cat corpora/GermEval/NER-de-test.tsv  | grep -v "^[#]" | cut -f2,3 |  sed "s/[^ \t]\+\(deriv\|part\)$/O/g" > corpora/GermEval/NER-de-test.tsv.conv
cat corpora/GermEval/NER-de-dev.tsv  | grep -v "^[#]" | cut -f2,3|  sed "s/[^ \t]\+\(deriv\|part\)$/O/g" > corpora/GermEval/NER-de-dev.tsv.conv
```


For the training we use the German Wikipedia embeddings from the [Facebook Research group](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). The embeddings can be quite large (above 10GB), especially, as the files will be decompressed. These (and all other embeddings) can be downloaded with the following command:

```
python3 download_model_embeddings.py all_models
```

If you want to train on a different language, you can also check if there are pre-computed embeddings available [here](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). To compute new embeddings you can follow the [manual from fastText](https://github.com/facebookresearch/fastText).

Next, the configuration needs to be edited. First, we create a directory where the model will be stored:

```
mkdir model_germeval
```

Then, we create the configuration file. For this, we use the configuration template ( [config.template](https://github.com/riedlma/sequence_tagging/blob/master/config.template)) and copy it to the model folder:

``` 
cp config.template model_germeval/config
```

At least, all parameters that have as value *TODO* need to be adjusted. Using the current setting, we adjust following parameters (a more detailed description of the configuration is found [here](#parameters-in-the-configuration-file)):

```
[PATH]
#path where the model will be written to, $PWD refers to the directory where the configuration file is located
dir_model_output = $PWD

...
filename_train = corpora/GermEval/NER-de-train.tsv.conv 
filename_dev =   corpora/GermEval/NER-de-dev.tsv.conv 
filename_test =  corpora/GermEval/NER-de-test.tsv.conv 
... 

[EMBEDDINGS]
# dimension of the words
dim_word = 300
# dimension of the characters
dim_char = 100
# path to the embeddings that are used
filename_embeddings = ./embeddings/wiki.de.bin
# path where the embeddings defined by train/dev/test are written to
filename_embeddings_trimmed = ${PATH:dir_model_output}/wiki.de.bin.trimmed.npz
...

```

Before we train the model, we build a matrix of the embeddings that are contained in the train/dev/test in addition to the vocabulary, with the *build_data.py* script. For training and testing only these smaller embeddings (specified with in the config with *filename_embeddings_trimmed*) are required. The larger ones (specified with *filename_embeddings*) can be deleted.

```
python3 build_data.py model_germeval/config
```

If you want to apply the model to other vocabulary then the one specified in train/dev/test, the model will not have any word representation and will mainly rely on the character word embedding. To prevent this, the easiest way is to add them in the CoNLL format as further parameters to the *build_data.py* script:

```
python3 build_data.py model_germeval/config vocab1.conll vocab2.conll
```


After that step, the new model can be trained, using the following command: 

```
python3 train.py model_germeval/config
```

The model can be applied to e.g. the test file as follows:

```
python3 test.py model_germeval/config corpora/GermEval/NER-de-test.tsv.conv
```



## Transfer Learning

For performing the transfer learning you first need to train a model e.g. based on the GermEval data as described [here](#train-a-new-model). Be aware, that you added the vocabulary and the tagsets when training the basic model. If you want to perform transfer learning you might want to copy the directory as otherwise the further learning steps will replace the previous model. Take care to adjust the * dir_model_output* value within the configuration file. The easiest way is to add them as additional parameters, when building the vocabulary, e.g.:

```
python3 build_data.py model_germeval/config transfer_training.conll transfer_dev.conll test_transfer.conll
```

However, this step needs to be performed before training the model. If you have already trained a model you would need to re-train a model with the additional vocabulary. 

Whereas there is not explicit parameter fitting to these words, in this way the embeddings will be available for the model. 

After the model has been trained the transfer learning step can be accomplished with the *transfer_learning.py* script, that expects the following parameters:

```
python transfer_learning.py configuration transfer_training.conll transfer_dev.conll
```

After the training, new text files in the domain of the transfer learning files as described [here](#predict-labels-for-new-text).


## Predict Labels for New Text

To test a model, the *test.py* script is used and expects, the configuration file of the model and the test file

``` 
python3 test.py model_configuration test_file.conll
```

The test script has further parameters in order to process several test files, different formats and write to output files directly. By calling the script with the *-h* argument, these will be shown:

```
python3 test.py -h

usage: test.py [-h] [-i {SYSTEM,FILE}] [-o {SYSTEM,FILE}] [-of OUTPUT_FOLDER]
               [-f {CONLL,TEXT,TOKEN}]
               config_file [test_files [test_files ...]]

positional arguments:
  config_file
  test_files

optional arguments:
  -h, --help            show this help message and exit
  -i {SYSTEM,FILE}, --input {SYSTEM,FILE}
                        if FILE is selected the file has to be passed as
                        additional parameter. If SYSTEM is selected the input
                        will be read from the standard input stream.
  -o {SYSTEM,FILE}, --output {SYSTEM,FILE}
                        if FILE is selected an output folder needs to be
                        specified (-of). If SYSTEM is selected, the standard
                        output stream will be used for the output.
  -of OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
  -f {CONLL,TEXT,TOKEN}, --format {CONLL,TEXT,TOKEN}
```

It is possible to read input that should be tagged from the standard input stream (-i SYSTEM) or from files. Furthermore, the output can be either written to the standard output or to files. If no parameter is specified, it will be written to the standard output stream. Otherwise, it will be written to a file with the name of the test file. In order to prevent ot override existing files, we advise to create and specify an output folder with the parameter *-of*. 
Currently, we support the files to be in the CoNLL format, in token format (words are tokenized and there is one sentence per line) or in plain text format (no tokenization), as described in the Table below:

| Format | Example | Description|
|--------|---------|------------|
| CONLL  | In O <br>Madrid B-LOC<br>befinden O<br> sich O<br> Hochschulen O<br>, O<br>Museen O<br>und O<br>Kultureinrichtungen O<br>. O | As CONLL format<br>we expect the files<br>to contain the token<br>in the first column.<br>All remaining columns will <br>be ignored.|
| TOKEN  | In Madrid befinden sich Hochschulen , Museen und Kultureinrichtungen .| The text is tokenized by whitespaces|
| TEXT  | In Madrid befinden sich Hochschulen, Museen und Kultureinrichtungen.| The text is not tokenized by whitespaces|


For the plain text format, nltk is required. It can be installed as follows:

```
pip3 install pip3
python3
import nltk
nltk.download('punkt')
exit()
``` 

## Server for Predicting Labels for New Text

If you want to use the NER tool as a service you can start a web server that gives responses to the given queries. For this you can specify a port (e.g. *-p 10080*) and a model configuration, e.g.:

```
python3 test_server.py -p 10080 model_configuration
```

The server processes two arguments: *text* expects the document for which named entity labels should be predicted. With the optional argument *format*, the input format can be specified (CONLL, TEXT, TOKEN). Further information about these formats is given [here](#predict-labels-for-new-text). We will show examples for each of the formats in the table below for the sentence: *Die Hauptstadt von Spanien ist Madrid*


| Format | Example |
|--------|---------|
| CONLL  | curl "localhost:10080?format=CONLL&text=Die%20O%0AHauptstadt%20O%0Avon%20O%0ASpanien%20O%0Aist%20O%0AMadrid%20O%0A.%20O%0A" |
| TOKEN | curl "localhost:10080?format=TOKEN&text=Die%20Hauptstadt%20von%20Spanien%20ist%20Madrid%20." |
| TEXT | curl "localhost:10080?format=TEXT&text=Die%20Hauptstadt%20von%20Spanien%20ist%20Madrid." |




## Parameters in the Configuration File

The configuration file is divided in three sections. The section *PATH* contains all variables that specify the locations of the model and labeled data. The *EMBEDDINGS* section contains all parameters for the word embeddings and the *PARAM* section contains all further parameters for the machine learning as well as pre-processing.

```
[PATH]
#path where the model will be written to
dir_model_output = $PWD
dir_vocab_output = ${dir_model_output}
dir_model = ${dir_model_output}/model.weights/
path_log = ${dir_model_output}/test.log


filename_train = TODO
filename_dev =   TODO
filename_test =  TODO

# these are the output paths for the vocabulary, the 
# tagsets and the characters used in the train/dev/test set
filename_words = ${dir_vocab_output}/words.txt
filename_tags = ${dir_vocab_output}/tags.txt
filename_chars = ${dir_vocab_output}/chars.txt


[EMBEDDINGS]
# dimension of the words
dim_word = 300
# dimension of the characters
dim_char = 100
# path to the embeddings that are used 
filename_embeddings = TODO
# path where the embeddings defined by train/dev/test are written to
filename_embeddings_trimmed =  ${PATH:dir_model_output}/embeddings.npz 
# models can also be trained with random embeddings that are 
# adjusted during training
use_pretrained = True
# currently we support: fasttext, glove and w2v
embedding_type = fasttext
# if using embeddings larger than 2GB this option needs to be switched on
use_large_embeddings = False
# number of embeddings that are dynamically changed during testing
oov_size = 0


# here, several parametesr of the machine learning and pre-processing
# can be changed
[PARAM]
lowercase = True
max_iter = None
train_embeddings = False
nepochs = 15
dropout = 0.5
batch_size = 20
lr_method = adam
lr = 0.001
lr_decay = 0.9
clip = -1
nepoch_no_imprv = 3
hidden_size_char = 100
hidden_size_lstm = 300
use_crf = True
use_chars = True

```





## Citation


If you use this model cite the source code of [Guillaume Genthial](https://github.com/guillaumegenthial/sequence_tagging). If you use the German model and the extension, you can cite our paper:

```
@inproceedings{riedl18:_named_entit_recog_shoot_german,
  title = {A Named Entity Recognition Shootout for {German}},
  author = {Riedl, Martin and Padó, Sebastian},
  booktitle = {Proceedings of Annual Meeting of the Association for Computational Linguistics},
  series={ACL 2018},
  address = {Melbourne, Australia},
  note = {To appear},
  year = 2018
}

```


## License

This project is licensed under the terms of the Apache 2.0 ASL license (as Tensorflow and derivatives). If used for research, citation would be appreciated.



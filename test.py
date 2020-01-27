#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 13:52:43 2017

@author: riedlmn
"""

import sys,os
import argparse
from model.ner_model import NERModel
from model.data_utils import FileStream, FileFormat
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_embedding_vectors, get_processing_word, add_oov_words
#import json

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
parser.add_argument("test_files",nargs='*')
parser.add_argument("-i" ,"--input", choices=["SYSTEM","FILE"],default="FILE",help="if FILE is selected the file has to be passed as additional parameter. If SYSTEM is selected the input will be read from the standard input stream.")
parser.add_argument("-o" ,"--output", choices=["SYSTEM","FILE"],default="SYSTEM",help="if FILE is selected an output folder needs to be specified (-of). If SYSTEM is selected, the standard output stream will be used for the output.")
parser.add_argument("-of" ,"--output_folder",default=".")
parser.add_argument("-f" ,"--format", choices=["CONLL","TEXT","TOKEN"],default="CONLL")

args = parser.parse_args()

if args.input=="SYSTEM" and len(args.test_files)>0:
    sys.stderr.write("Files for predicting annotations have been specified. However, as the STDIN option has been specified they will not be tagged\n")
if args.input=="FILE" and len(args.test_files)==0:
    sys.stderr.write("No files are specified to be tagged!\n")
    sys.exit(0)


input_format = FileStream.FILE
if args.input == "SYSTEM":
    input_format = FileStream.SYSTEM
    

output_format = FileStream.SYSTEM
if args.output == "FILE":
    output_format = FileStream.FILE


file_format = FileFormat.CONLL
if args.format =="TEXT":
    file_format = FileFormat.TEXT
elif args.format =="TOKEN":
    file_format = FileFormat.TOKEN

    
config_file = args.config_file
test_filenames = args.test_files
    
config = Config(config_file)

#check if some test file is directory and add files from the directory
new_test_filenames = []
for f in test_filenames:
    if os.path.isdir(f):
        for fi in os.listdir(f):
            new_test_filenames.append(os.path.join(f,fi))
    else:
        new_test_filenames.append(f)
test_filenames=new_test_filenames

# build model
conll_test_files = []
for f in test_filenames:
    test  = CoNLLDataset(f, config.processing_word,None, config.max_iter, 
                         stream=input_format,file_format=file_format)
    conll_test_files.append(test)
#add OOV words to the model
add_oov_words(conll_test_files,config)

model = NERModel(config)
model.build()
model.restore_session(config.dir_model)


# create dataset
i = 0
for test in conll_test_files:
    output_target = sys.stdout
    if output_format==FileStream.FILE:
        if input_format== FileStream.SYSTEM:    
            output_file = os.path.join(args.output_folder,"output.txt")
            output_target=open(output_file,"a")
            sys.stderr.write("output is appended to :"+output_file+"\n")
        else:
            output_target=open(os.path.join(args.output_folder,os.path.basename(test_filenames[i])),"w")
    model.predict_test(test,output=output_target)
    i+=1

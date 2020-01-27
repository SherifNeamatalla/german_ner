#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:40:34 2017

@author: riedlmn
"""

import configparser
import sys,os
import shutil, errno


DIR = "SOME_PATH"
test = {}

train = {}
train["germeval"]= DIR+"/NER/corpora/GermaNER-train.conll"
train["conll2003"] =  DIR+"/NER/corpora/new/utf8.deu.mr.train"
train["lft"]=  DIR+"/NER/corpora/tokenized/enp_DE.lft.mr.tok.train.bio"
train["onb"]=  DIR+"/NER/corpora/tokenized/enp_DE.onb.mr.tok.train.bio"
train["sbb"]=  DIR+"/NER/corpora/tokenized/enp_DE.sbbmr.mr.tok.train.bio"



test = {}
test["germeval.test"]= DIR+"/NER/corpora/GermaNER-test.conll"
test["conll.test"] =  DIR+"/NER/corpora/new/utf8.deu.mr.testb"
test["lft.test"]=  DIR+"/NER/corpora/tokenized/enp_DE.lft.mr.tok.test.bio"
test["onb.test"]=  DIR+"/NER/corpora/tokenized/enp_DE.onb.mr.tok.test.bio"
test["sbb.test"]=  DIR+"/NER/corpora/tokenized/enp_DE.sbbmr.mr.tok.test.bio"

test["germeval.dev"]= DIR+"/NER/corpora/GermaNER-dev.conll"
test["conll.dev"] =  DIR+"/NER/corpora/new/utf8.deu.mr.testa"
test["lft.dev"]=  DIR+"/NER/corpora/tokenized/enp_DE.lft.mr.tok.dev.bio"
test["onb.dev"]=  DIR+"/NER/corpora/tokenized/enp_DE.onb.mr.tok.dev.bio"
test["sbb.dev"]=  DIR+"/NER/corpora/tokenized/enp_DE.sbbmr.mr.tok.dev.bio"

embeddings = {}
#embeddings["german_de_100"]= (100, DIR+"/fastText/German_de_docs_skip_100.vec")
#embeddings["german_de_500"]= (500, DIR+"/fastText/German_de_docs_skip_500.vec")
#embeddings["de_100"]= (500, DIR+"/w2v/de_txt.s500.n15.skip")
#embeddings["wiki_500"] = (500, DIR+"/NER/sequence_tagging3/de_tokenized_clean_w2v_skip_w5_n5_s500d.txt")
embeddings["facebook_de"]= (300, DIR+"/embeddings/fasttext/wiki.de.bin")
embeddings["German_de_300"]=(300, DIR+"/fastText/German_de_docs_skip_300.bin")
#embeddings["mr_wiki_de_300"]= DIR+"/fastText/de_wiki.txt.seg.10.lower.txt.300.bin"
#embeddings["mr_wiki_de_upper_300"]= DIR+"/fastText/de_wiki.txt.seg.10.txt.300.bin"


configfile = sys.argv[1]


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise
        
        
param = configparser.SafeConfigParser()
param.read(configfile)

for train_key,train_val in train.items():
    for trans_key,trans_val in train.items():
        for emb_key,(emb_dim,emb_val) in embeddings.items():
            out_dir = "transfer3/transfer_train_"+train_key+"_"+emb_key+"_"+trans_key
            old_out_dir = "train3_"+train_key+"_"+emb_key
            #out_dir = "train_"+train_key+"_"+emb_key
            
            #if not os.path.exists(out_dir):
            #    print( "#copy")
            #    copyanything(old_out_dir,out_dir)
            #else:
            #    continue
            #param.set("PARAM","use_chars","False")
            param.set("PATH","dir_model_output",out_dir)
            param.set("PATH","dir_vocab_output",out_dir)
            param.set("PATH","path_log",out_dir+ "/test.log")
            param.set("PATH","filename_train",train_val)
            param.set("PATH","filename_dev","all_train_test_dev.bio")
            param.set("EMBEDDINGS","dim_word",str(emb_dim))
            param.set("EMBEDDINGS","filename_glove",emb_val)
            param.set("EMBEDDINGS","filename_trimmed",emb_val+".trimmed.npz")
            new_config_file = out_dir+"/config_"+train_key+"_"+emb_key
            param.write(open(new_config_file,"w"))
            #print("python build_data.py "+ new_config_file)
            print("python transfer_learning.py %s %s %s " %(new_config_file,trans_val,param.get("PATH","filename_dev")))
            for (t,tf) in test.items():
                #if not os.path.exists(out_dir+ "/"+t+".res"):
                    print("python test.py "+new_config_file+ " "+tf + " > "+out_dir+ "/"+t+".res")
    
            for (t,tf) in train.items():
                #if not os.path.exists(out_dir+ "/"+t+".res"):
                    print("python test.py "+new_config_file+ " "+tf + " > "+out_dir+ "/"+t+".train"+".res")
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 20:09:33 2018

@author: riedlmn
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import sys,os
import argparse

from model.ner_model import NERModel
from model.data_utils import FileStream, FileFormat
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_embedding_vectors, get_processing_word, add_oov_words



parser = argparse.ArgumentParser()
parser.add_argument("config_file")
#parser.add_argument("-f" ,"--format", choices=["CONLL","TEXT","TOKEN"],default="CONLL")
parser.add_argument("-p" ,"--port", type = int,default="10080")

args = parser.parse_args()

port = args.port

    
config_file = args.config_file
config = Config(config_file)

# load model
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)

class Serv(BaseHTTPRequestHandler):
    
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
    def do_GET(self):
        self._set_headers()
        query_components = parse_qs(urlparse(self.path).query)
        text = ""          
        text = query_components["text"][0]
        file_format = FileFormat.TEXT
        if "format" in query_components:
            
            if query_components["format"][0] =="CONLL":
                file_format = FileFormat.CONLL
            elif query_components["format"][0] =="TEXT":
                file_format = FileFormat.TEXT
            elif query_components["format"][0] =="TOKEN":
                file_format = FileFormat.TOKEN
        test  = CoNLLDataset(text, config.processing_word,
                     None, config.max_iter,file_format = file_format, stream = FileStream.DIRECT)
        model.predict_test(test,output=self.wfile,write_binary=True)
       
   
def run(server_class=HTTPServer, handler_class=Serv, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print ('Starting httpd using port '+str(port))
    httpd.serve_forever()
run(port=port)
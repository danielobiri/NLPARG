#!/usr/bin/env python3
import pandas as pd
from Bio import SeqIO
from nltk import trigrams
import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import pickle
import random
import os
import sys
from flask import Flask, jsonify, request
import argparse
import datetime

random.seed(12345)
os.environ['PYTHONHASHSEED']='0'
seed=42
np.random.seed(seed)
ts=datetime.datetime.now()

sequence_index = []
def preprocess(args):
    query_seq=[]
    
    global sequence_index
    
    query_fasta=args.input
    number_AMR=args.Num_AMRG
    type_vector_num=args.type_vector_num
    if type_vector_num==1:
       type_vector=r"Glove_model.h5"
    if type_vector_num==2:
       type_vector=r"SG_model.h5"
    if type_vector_num==3:
       type_vector=r"CBOW_model.h5"
    else:
       print(" enter a number to represent any of the following, 1:Glove_model.h5,2:SG_model.h5,3:CBOW_model.h5")
      
    for index,seque in enumerate(SeqIO.parse(query_fasta,'fasta')):
        sequence_index.append(index)
        tri_tokens = trigrams(seque.seq)
        temp_str = ""
        for item in ((tri_tokens)):
 
            temp_str = temp_str + " " +item[0] + item[1] + item[2]
        #print (temp_str)
        query_seq.append(temp_str.strip())
    print(len(query_seq))
    
    MAX_SEQUENCE_LENGTH = 450
    with open('tokenizer.', 'rb') as handle:
        tokenizer = .load(handle)
    sequences = tokenizer.texts_to_sequences(query_seq)
    data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
    
    loaded_1 = keras.models.load_model(type_vector)
    o_pred=loaded_1.predict(data_test)
    
    print("You submitted %i sequences for testing" %len(o_pred))
    
   
    out_handle = open(args.input+'__results', 'w')
    out_handle.write("Author: Daniel Ananey-Obiri (dananeyobiri@gmail.com)\n" + str(ts) + "\n-----------------------------\nYour choice of embedding layer: " + str(type_vector) + "\ncontext window size: 5 \nk-mer size:3 \n-----------------------------\n")

    print ("Creating result file")
    count = 0
    for index,sequence in enumerate(SeqIO.parse(query_fasta, 'fasta')):
        if index in sequence_index:
            out_handle.write(">%s|%s\n%s\n" % (sequence.description, 
                            str("This is %s with a probability of %.2f" %("an antimicrobial resistance gene sequence" if \
                                                                             o_pred[count][0] > 0.5 else \
                                                                             " a non-antimicrobial resistance gene sequence",o_pred[count][0])), sequence.seq))
            count = count + 1

    file=open(args.input + '__results',"r")
    data=file.read()
    occurrences=data.count("an antimicrobial resistance gene sequence")
    
    #data.insert(1,'\n' + number_AMR + str(occurrences) + " out of " + str(len(o_pred)))
    #file.close()

    out_handle.write('\n' + number_AMR + str(occurrences) + " out of " + str(len(o_pred)))
    
    file.close()

    
    out_handle.close()
    print ("Successfully executed")
    return out_handle


def main():
    parser=argparse.ArgumentParser(prog='NLP_ARG.py')
    parser=argparse.ArgumentParser(description="Predict antimicrobial resistance of sequence")
    parser.add_argument("-choice_wordvector",dest="type_vector_num",help="select the kind of word vector (1=Global vector,2:continuous-bag-of-words,3: Skip-gram model)",required=True,type=int, \
         choices=range(1,4))
    parser.add_argument("-in",help="fasta input file must be fasta or text",dest="input",required=True,type=str)
    parser.add_argument("-out",help="fasta output with probabilities and annotation",dest='output',type=str,required=True)
    parser.add_argument("-number_AMR",help="number antimicrobial resistance genes sequences",dest='Num_AMRG',type=str,default="The number of Antimicrobial resistance sequences are ")

    parser.set_defaults(func=preprocess)
    args=parser.parse_args()
    args.func(args)

   
if __name__ == '__main__':
    main()


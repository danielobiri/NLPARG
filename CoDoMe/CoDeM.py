#!/usr/bin/env python3


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
from nltk import bigrams
import gensim, logging
import numpy as np
from nltk import trigrams
#from bert import tokenization
from keras import backend as K
#pip install --upgrade tensorflow
import tensorflow as tf
import keras
import pydot as pyd
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pyd
import matplotlib.pyplot as plt
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Dense, Embedding, LSTM, GRU
from keras.initializers import Constant
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector,Dropout, Conv1D, MaxPooling1D, UpSampling1D,GlobalMaxPool1D
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from keras.preprocessing.sequence import pad_sequences
import pickle
import random
random.seed(12345)
import os
os.environ['PYTHONHASHSEED']='0'
seed=42
np.random.seed(seed)
from termcolor import colored
from colorama import Fore, Back, Style
import sys
from flask import Flask, jsonify
import pickle
import numpy as np
from flask import Flask, request
import argparse

import datetime

import time
ts=datetime.datetime.now()
# In[7]:

sequence_index = []
def preprocess(args):
    query_seq=[]
    
    global sequence_index
    
    input_fastq=args.input
    type_vector_num=args.type_vector_num
    if type_vector_num==1:
       type_vector=r"GloVe_CoDeM.h5"
    if type_vector_num==2:
       type_vector=r"Skipgram_CoDeM.h5"
    if type_vector_num==3:
       type_vector=r"CBOW_CoDeM.h5"
    if type_vector_num==4:
       type_vector=r"FastText_CoDeM.h5"
    else:
        print(" enter a number to represent any of the following, 1:Glove_model.h5,2:SG_model.h5,3:CBOW_model.h5,4:FastText")
    
    trig_sequence= []
    seq_index=[]

    for index, record in enumerate(SeqIO.parse(input_fastq,'fastq')):
            seq_index.append(index)
                tri_tokens = trigrams(record.seq)
                    temp_str = ""
                        for item in ((tri_tokens)):
                                    temp_str = temp_str + " " +item[0] + item[1] + item[2]
                                        text_unc.append(temp_str.strip())
                                        print(len(trig_sequence))


    print("---------------------------------------------------------------------------"+\n+" Contaminant Determining Model (CoDeM)" +\n +" Author: Daniel ananey-Obiri, PhD " +\n + "North Carolina Agricultual and Technical State University")
    

    MAX_SEQUENCE_LENGTH = 300
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(trig_sequence)
    data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
    
    loaded_1 = keras.models.load_model(type_vector)
    l_pred=loaded_1.predict(data_test)
    print("You submitted %i sequences for testing" %len(l_pred))
    print("Uncontaminated reads: %i" %len(l_pred[l_pred<0.5]))
    print("Contaminated reads: %i" %len(l_pred[l_pred > = 0.5]))


    uncontaminated_sam=open(filename +str("_")+'uncontaminateds','w+')
    contaminated_sam=open(filename + str("_")+ 'contaminateds','w+')

    count = 0
    for index, record in enumerate(SeqIO.parse(trig_sequence,'fastq')):
            if index in seq_index :
                        if l_pred[count]>=0.5:

                                        uncontaminated_sam.write("%s" %(record.format("fastq-sanger")))
                                                elif l_pred[count]<0.5 :
                                                                contaminated_sam.write("%s" %(record.format("fastq-sanger")))
                                                                        count = count + 1


    uncontaminated_sam.close()
    contaminated_sam.close()


def main():
    parser=argparse.ArgumentParser(prog='CoDeM.py')
    parser=argparse.ArgumentParser(description="Identify contaminated genomic reads")
    parser.add_argument("-choice_wordvector",dest="type_vector_num",help="select the kind of word vector (1=Global vector,2:continuous-bag-of-words,3: Skip-gram model, 4:FastText)",required=True,type=int, \
         choices=range(1,5))
    parser.add_argument("-in",help="fasta input file must be fasta or text",dest="input",required=True,type=str)

    parser.set_defaults(func=preprocess)
    args=parser.parse_args()
    args.func(args)

   
if __name__ == '__main__':
    main()


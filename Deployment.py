# coding: utf-8

# In[270]:
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

#tf.set_random_seed(12345)
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Dense, Embedding, LSTM, GRU
from keras.initializers import Constant
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector,Dropout, Conv1D, MaxPooling1D, UpSampling1D,GlobalMaxPool1D
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
import seaborn as sns
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from keras.models import Sequential
from keras.layers import Embedding, Dense, Embedding, LSTM, GRU
from keras.initializers import Constant
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Dense, Lambda, LSTM, RepeatVector,Dropout, Conv1D, MaxPooling1D, UpSampling1D,GlobalMaxPool1D
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
import seaborn as sns
from scipy import interp
from sklearn.metrics import roc_curve, auc

# In[74]:
#glove_input_file = r"New_gloves_CW_4.txt"
#word2vec_output_file = 'glove_to_word2vec.txt'
#glove2word2vec(glove_input_file, word2vec_output_file)
new_model = gensim.models.Word2Vec.load("CBOW_model_three")


#new_model =KeyedVectors.load_word2vec_format('glove_to_word2vec.txt',binary=False)



text= []
for index, record in enumerate(SeqIO.parse(r"Pre_met_Positive_training_set.fasta",'fasta')):
    tri_tokens = trigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        #print(item),
        temp_str = temp_str + " " +item[0] + item[1] + item[2]
    #print (temp_str)
    text.append(temp_str.strip())
print(len(text))





#ensure that all training sequences (text) have the same length using pad
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 450
MAX_NB_WORDS = len(new_model.wv.vocab)
EMBEDDING_DIM = 200

tokenizer=tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_NB_WORDS, lower=True
)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

#shape is (num_samples, num_timesteps=the longest text or sequence)
print('Shape of data tensor:',data.shape)




print('Preparing embedding matrix')

embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))

print (embedding_matrix.shape)

for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    if word.upper() in new_model.wv.vocab:
        embedding_vector = new_model[word.upper()]
        embedding_matrix[i] = embedding_vector

      
      
      
num_words_plus=len(word_index) + 1
#model=Sequential()
embedding_layer = Embedding(num_words_plus,
                        EMBEDDING_DIM,
                        #embeddings_initializer=Constant(embedding_matrix),
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False)


from keras.models import Model
def rnn_architecture():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), )
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.1, return_sequences = True))(embedded_sequences)
    x = Bidirectional(LSTM(32, dropout=0.5,recurrent_dropout=0.1))(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy']
             )
    return model



      
     
print("using saved model")
# Option 1: Load with the custom_object argument.
loaded_1 = keras.models.load_model("Glove_model.h5")











labels = np.vstack((np.ones((1000, 1)),
                    np.zeros((1000,1))))
labels_one_dim = labels.reshape(labels.shape[0], )

# In[280]:


print('Preparing embedding matrix')

embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))

print (embedding_matrix.shape)

for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    if word.upper() in new_model.wv.vocab:
        embedding_vector = new_model[word.upper()]
        embedding_matrix[i] = embedding_vector
print(embedding_matrix[10])
print(len(embedding_matrix))
print(new_model.wv.most_similar('LDI'))


# In[281]:




num_words_plus=len(word_index) + 1
#model=Sequential()
embedding_layer = Embedding(num_words_plus,
                        EMBEDDING_DIM,
                        #embeddings_initializer=Constant(embedding_matrix),
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False)


from keras.models import Model
def rnn_architecture():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), )
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.1, return_sequences = True))(embedded_sequences)
    x = Bidirectional(LSTM(32, dropout=0.5,recurrent_dropout=0.1))(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer= 'adam',
                  metrics=['accuracy']
             )
    return model



model_arc=rnn_architecture()
model_arc.summary()


# Option 1: Load with the custom_object argument.
loaded_1 = keras.models.load_model("Glove_model.h5")

if __name__ == '__main__':
   
    query_fasta = sys.argv[1] 

    query_seq=[]
    keep_track_index = []
    for index,seque in enumerate(SeqIO.parse(query_fasta,'fasta')):
        keep_track_index.append(index)
        tri_tokens = trigrams(seque.seq)
        temp_str = ""
        for item in ((tri_tokens)):
            #print(item),
            temp_str = temp_str + " " +item[0] + item[1] + item[2]
        #print (temp_str)
        query_seq.append(temp_str.strip())
    print(len(query_seq))

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(query_seq)
    data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")
    
    
  
    o_pred=loaded_1.predict(data_test)
    o_pred=np.where(o_pred > 0.5,1,0)
    print("You submitted %i sequences for testing" %len(o_pred))
 
    
    
    
    out_handle = open(query_fasta+'_results', 'w')

    print ("Creating result file")
    count = 0
    for index, record in enumerate(SeqIO.parse(query_seq, 'fasta')):
        if index in keep_track_index:
            out_handle.write(">%s|%s\n%s\n" % (record.description, 
                            str(o_pred[count][0]), 
                            record.seq))
            count = count + 1

    out_handle.close()
    print ("Done!! ")
    
    


    
    



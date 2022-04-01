#!/usr/bin/env python
# coding: utf-8

# In[120]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 7.0)
#from gensim.scripts.glove2word2vec import glove2word2vec
#from gensim.models import KeyedVectors

import math
import numpy as np

seed = 42
np.random.seed(seed)

import os
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
from nltk import bigrams
from nltk import trigrams

import gensim
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.metrics import matthews_corrcoef

from keras.layers import Dropout
from keras.layers import Input, Dense, Lambda, LSTM
from keras.models import Model
from keras import backend as K
#from keras import objectives
from keras.datasets import mnist
from keras import regularizers
from keras.layers import GaussianNoise
from keras.layers import Activation
from keras.callbacks import LearningRateScheduler, EarlyStopping

#from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import regularizers
#from keras.regularizers import l2, activity_l2, l1, activity_l1
#from keras.optimizers import Adam, SGD

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


import dask.dataframe as dd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy import interp
from itertools import cycle
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB

#new_model = gensim.models.Word2Vec.load('word2vec_model_current')



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#glove_input_file = r"C:\Users\AMD\Desktop\New folder\New_gloves.txt"
#word2vec_output_file = 'glove_to_word2vec.txt'
#glove2word2vec(glove_input_file, word2vec_output_file)
#new_model =KeyedVectors.load_word2vec_format('glove_to_word2vec.txt',binary=False)
new_model = gensim.models.Word2Vec.load('word2vec_model_current')
temp_word = np.zeros(shape=(2416,200))
seq=[]
for index, record in enumerate(SeqIO.parse(r"Pre_Metal_Training_Old.fasta",'fasta')):
    #print(record.seq)
    sum_of_sequence = 0
    tri_tokens = trigrams(record.seq)   
    for item in ((tri_tokens)):
        tri_str = item[0] + item[1] + item[2]
        if tri_str not in list(new_model.wv.vocab):
            continue
        sum_of_sequence = sum_of_sequence + new_model[tri_str.strip()]  
    temp_word[index] = sum_of_sequence/len(sum_of_sequence)
print(len(temp_word))
    
print(temp_word.shape)
 


#print(new_model['MRI'])
y_temp_word = np.vstack((np.zeros((1208, 1)), 
                    np.ones((1208,1))))
#Standardize your data


scaled_data=temp_word

scaler = preprocessing.StandardScaler()
scaled_data=scaler.fit(temp_word)
scaled_data=scaled_data.transform(temp_word)
#print(scaled_data)

#temp_scaler = preprocessing.StandardScaler().fit(temp_word)
#temp_word_scaled = temp_scaler.transform(temp_word)

c, r = y_temp_word.shape
y_temp_word = y_temp_word.reshape(c,)
#print(y_temp_word)


# In[121]:


from sklearn.decomposition import PCA
pca=PCA(10)

#fit data
pca.fit(scaled_data)
print(pca.components_)
print(pca.explained_variance_ratio_)
scaled_data=pca.transform(scaled_data)
scaled_data


# In[122]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from numpy import concatenate


from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn import model_selection
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.semi_supervised import LabelPropagation
import numpy as np


# In[123]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import make_circles


# In[124]:


# generate ring with inner box
n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)
outer, inner = 0, 1
labels = np.full(n_samples, -1.)

labels[0] = outer


labels[-1] = inner

# Learn with LabelSpreading
label_spread = LabelSpreading(kernel='knn', alpha=0.85)
label_spread.fit(X, labels)


# In[125]:


# Plot output labels
output_labels = label_spread.transduction_
plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1)

#outer
plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color='navy',
            marker='s', lw=0, label="outer labeled", s=10)

#Inner
plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color='c',
            marker='s', lw=0, label='inner labeled', s=10)

#unlabeled
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange',
            marker='.', label='unlabeled')
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Raw data (2 classes=outer and inner)")

plt.subplot(1, 2, 2)
output_label_array = np.asarray(output_labels)
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]
plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',
            marker='s', lw=0, s=10, label="outer learned")
plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',
            marker='s', lw=0, s=10, label="inner learned")
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Labels learned with Label Spreading (KNN)")

plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
plt.show()



"""# Plot output labels



plot_outer, = plt.plot(X[outer_numbers, 0], X[outer_numbers, 1], 'rs')
plot_inner, = plt.plot(X[inner_numbers, 0], X[inner_numbers, 1], 'bs')
plt.legend((plot_outer, plot_inner), ('Outer Learned', 'Inner Learned'),
           'upper left', numpoints=1, shadow=False)
plt.title("Labels learned with Label Spreading (KNN)")

plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)"""


# In[126]:


n_seqs=2416
indices = np.arange(n_seqs)
np.random.shuffle(indices)
X = scaled_data[indices]
y = y_temp_word[indices]

n_tr = int(n_seqs * 0.4139)
n_va = int(n_seqs * 0.1)
n_te = n_seqs - n_tr - n_va
X_train = X[:n_tr]
y_train = y[:n_tr]
X_valid = X[n_tr:]
y_valid = y[n_tr:]

X_train.shape


# In[127]:


scaled_data=X_train
y_temp_word=y_train


# In[128]:



# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


precision_scores_mean_list = []
recall_scores_mean_list = []
f1_scores_mean_list = []

scores=[]
accuracy_scores=[]
precision_scores=[]
recall_scores=[]
f1_scores=[]
matthews_score=[]


accuracy_scores=[]
mean_acc=[]
mean_pre=[]
mean_recall=[]
mean_f_1=[]
mean_MCC=[]
    

outer_random_seed_list = [3,13,23,33,43,53,63,73,83,93]
random_seed_list = [2,12,22,32,42,52,62,72,82,92]
#outer_random_seed_list = [3,13,23]
#random_seed_list = [2,12,22]
# outer_random_seed_list = [13,23,33,43,53,63,73,83,93]
# random_seed_list = [12,22,32,42,52,62,72,82,92]
for index, rand_seed_i in enumerate(random_seed_list):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
     # Use variation of KFold cross validation that returns stratified folds for outer loop in the CV.
        # The folds are made by preserving the percentage of samples for each class.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=outer_random_seed_list[index])
        # Iterator over the CVs
    for index,(train_indexes, test_indexes) in enumerate(skf.split(scaled_data, y_temp_word)):
        X_train,X_test = scaled_data[train_indexes], scaled_data[test_indexes]
        y_train,y_test = y_temp_word[train_indexes], y_temp_word[test_indexes]
        
        Y_init=y_train.copy()     
        
        
        print(len(y_train))
        subset = np.random.choice(y_train.size, 810, replace=False)

        y_train[subset] = -1
        #print(subset)
        unique, counts = np.unique(y_train, return_counts=True)
        print(dict(zip(unique, counts)))
        

        
        #pipe = Pipeline([('Propagation', LabelPropagation(kernel='rbf',gamma=1, max_iter=10))])
        pipe=LabelSpreading(kernel='knn', alpha=0.2, max_iter=50,n_neighbors=10,tol=1e-6)
        #pipe = Pipeline([('Propagation', LabelPropagation(kernel='rbf',gamma=1, max_iter=50,tol=1e3))])
        
        pipe.fit(X_train, y_train)
        #print(pipe)
        #output_labels_classification = pipe.transduction_
        #unique, counts = np.unique(output_labels_classification, return_counts=True)
        #print(dict(zip(unique, counts)))
        
        #print(accuracy_score(y_or, output_labels_classification))
        
        y_pred = pipe.predict(X_test)
        proba_=pipe.predict_proba(X_test)
        print(proba_[:,1])
    
        

        
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        matthews_score.append(matthews_corrcoef(y_test, y_pred))
        scores.append(pipe.score(X_test, y_test))
        
    mean_acc.append(np.mean(accuracy_scores))
    mean_pre.append(np.mean(precision_scores))
    mean_recall.append(np.mean(recall_scores))
    mean_f_1.append(np.mean(f1_scores))
    mean_MCC.append(np.mean(matthews_score))
    

        #Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
        
print(accuracy_score(y_test, y_pred))
     
print('Mean_Accuracy: %.4f Mean_Precision %.4f Mean_Recall: %.4f' %(np.mean(mean_acc), np.mean(mean_pre), np.mean(mean_recall)))
print('Mean_F1-measure', np.mean(mean_f_1))
print("Mean_mathews Coefficient score",np.mean(mean_MCC))
#results.write(np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores))


# In[129]:



y_test=pipe.predict(X_valid)
precision_s = precision_score(y_test, y_valid)
recall_s = recall_score(y_test,y_valid )
Accuracy_s=accuracy_score(y_test,y_valid)
F_1_s=f1_score(y_test,y_valid)
MCC=matthews_corrcoef(y_test,y_valid)
print("Accuracy :%f recall: %f precision: %f F-score :%f MCC :%f" %(Accuracy_s,recall_s,precision_s, F_1_s,MCC))


# In[130]:


unique, counts = np.unique(Y_init[subset], return_counts=True)
dict(zip(unique, counts))


# In[131]:


unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))


# In[132]:


model=LabelSpreading(kernel='knn', alpha=0.2, max_iter=50,n_neighbors=10,tol=1e-6)

model.fit(X_train, y_train)
y_pred=model.predict(X_train)


# In[ ]:





# In[ ]:





# In[133]:


unique, counts = np.unique(model.transduction_[subset], return_counts=True)
dict(zip(unique, counts))


# In[134]:


from sklearn.metrics import classification_report
print(classification_report(Y_init[subset], model.transduction_[subset]))


# In[ ]:





# In[135]:


precision_s = precision_score(Y_init[subset], model.transduction_[subset])
recall_s = recall_score(Y_init[subset], model.transduction_[subset])
print(recall_s)
precision_s


# In[136]:


from Bio import SeqIO
from nltk import bigrams
from nltk import trigrams

new_model = gensim.models.Word2Vec.load('word2vec_model_current')
temp_word = np.zeros(shape=(2416,200))
seq=[]
testing_data = np.zeros(shape=(455,200))
for index, record in enumerate(SeqIO.parse(r"Metal_Predi_testing_refined",'fasta')):
  
    sum_of_sequence = 0
    tri_tokens = trigrams(record.seq)
    tri_tokens=list(tri_tokens)
    #tri_tokens=tri_tokens[:600]
    
    for item in ((tri_tokens)):
        tri_str = item[0] + item[1] + item[2]
        if tri_str not in list(new_model.vocab):
            print(tri_str)
            continue
        sum_of_sequence = sum_of_sequence + new_model[tri_str]   
    testing_data[index] = sum_of_sequence/len(sum_of_sequence)
print(len(testing_data))    
scaler = preprocessing.StandardScaler()
scaled_testing=scaler.fit(testing_data)
testing_data=scaled_testing.transform(testing_data)


from sklearn.decomposition import PCA
pca=PCA(10)

#fit data
pca.fit(testing_data)
testing_data=pca.transform(testing_data)






y_test=model.predict(testing_data)
#precision_s = precision_score(y_test, y_temp_word)
#recall_s = recall_score(y_test, y_temp_word)
#print(recall_s)
print(y_test)
o_pred=np.where(y_test > 0.5,1,0)
print(len(y_test[y_test==1]))
len(y_test[y_test==0])


# In[ ]:


import numpy as np
testing_data = np.zeros(shape=(2,2))
row=[5,7]
arr = np.vstack([testing_data,row])
arr


# In[ ]:


arr = np.empty((0,3), float)
arr=np.append(arr,[[1,2,3]], axis=0)
arr=np.append(arr,[[1,2,14]], axis=0)
arr


# In[ ]:


empty_array = np.array([])

to_append = np.array([1, 2, 3])


combined_array = np.append(empty_array, to_append)
combined_array = np.append(combined_array, to_append)
combined_array


# In[ ]:


to_append = np.array([1, 2, 3])
empty_array =np.empty((0,3), float)
x=np.vstack((to_append,empty_array))
append=[2,7,9]
x=np.vstack((x,append))
x


# In[ ]:





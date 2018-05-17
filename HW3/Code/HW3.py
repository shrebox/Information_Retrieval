
# coding: utf-8

# In[1]:

#Import packages

import os
import nltk
import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from random import shuffle
import copy
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:

## Pre-processing

tokenizer = RegexpTokenizer(r'\w+') #Tokenizer
stemmer = SnowballStemmer('english') #Snowball Stemmer
stops = set(stopwords.words('english')) #Stopwords
for i in stops:
    STOPWORDS.add(i)


# In[3]:

cwd = os.getcwd() #Current Working Directory
folders_path = os.path.join(cwd, r"Dataset\20_newsgroups")
folders = os.listdir(folders_path) #List of folders


# In[4]:

count_to_file = {} #Dictionary that maps file no to file path
file_to_count = {} #Dictionary that maps file path to file no
count = 0

for root, _, files in os.walk(folders_path):
    for file in files:
        file_path = os.path.join(root, file)
        count += 1
        
        file_to_count[file_path] = str(count)
        count_to_file[str(count)] = file_path


# In[5]:

training_data = {} #Dictionary that stores list of training files for each class
test_data = {} #Dictionary that stores list of test files for each class
class_dic = {} #Dictionary that maps class no to class/folder path
cls_count = 0
ratio = 0.7 #Training Data split ratio

for root, _, files in os.walk(folders_path):
    if (len(files) == 0):
        continue
    
    class_dic[str(cls_count)] = root
    train_data_len = int(ratio * len(files))
    test_data_len = len(files) - train_data_len
    
    data = []
    
    for file in files:
        file_path = os.path.join(root, file)
        data.append(file_to_count[file_path])
    
    shuffle(data) #Randomly shuffle data to split into test and train
    
    training_data[str(cls_count)] = data[:train_data_len]
    test_data[str(cls_count)] = data[-test_data_len:]
    
    cls_count += 1


# In[6]:

word_dic = {} #Dictionary that stores vocabulary for each class
freq_dic = {} #Dictionary that stores the frequency of occurance of each word in the vocabulary for each class

for cls in class_dic:
    
    if cls not in word_dic:
        word_dic[cls] = {}
        freq_dic[cls] = {}
    
    files = training_data[cls]
    for file in files:
        
        file_path = count_to_file[file]
    
        lines = ""
        
        f = open(file_path, 'r')
        lines = f.read()
        
        words = tokenizer.tokenize(lines) #tokenize
        words = [word.strip('_ ').lower() for word in words if word not in STOPWORDS] #lowercase 
        words = [stemmer.stem(word) for word in words] #stemmer
        words = [word for word in words if word not in STOPWORDS and len(word) > 0] #stopword removal

        for word in words:
            if word not in word_dic[cls]:
                word_dic[cls][word] = set()
                freq_dic[cls][word] = 0
            word_dic[cls][word].add(file)
            freq_dic[cls][word] += 1
    
    print(len(word_dic[cls]))


# In[7]:

for cls in word_dic:
    for i in word_dic[cls]:
        word_dic[cls][i] = sorted(word_dic[cls][i]) #Sort the docIDs


# In[8]:

word_dic_complete = set() #Find the no. of words in the complete vocabulary (For smoothing denominator)

for cls in class_dic:
    for word in word_dic[cls].keys():
            word_dic_complete.add(word)
            
total_vocab_size = len(word_dic_complete)
print(total_vocab_size)


# In[9]:

# Store dictionaries as json files

fp = open(os.path.join(cwd, r'Dataset\count_to_file.json'), 'w+')
json.dump(count_to_file, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\file_to_count.json'), 'w+')
json.dump(file_to_count, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\training_data.json'), 'w+')
json.dump(training_data, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\test_data.json'), 'w+')
json.dump(test_data, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\class_dic.json'), 'w+')
json.dump(class_dic, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\word_dic.json'), 'w+')
json.dump(word_dic, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\freq_dic.json'), 'w+')
json.dump(freq_dic, fp, sort_keys=True)
fp.close()


# In[10]:

# Load dictionaries

fp = open(os.path.join(cwd, r'Dataset\count_to_file.json'), 'r')
count_to_file = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\file_to_count.json'), 'r')
file_to_count = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\training_data.json'), 'r')
training_dataa = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\test_data.json'), 'r')
test_data = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\class_dic.json'), 'r')
class_dic = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\word_dic.json'), 'r')
word_dic = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\freq_dic.json'), 'r')
freq_dic = json.load(fp)
fp.close()


# In[11]:

## NB Training

prior = {} #Dictionary that stores log of prior probability for each class 
cond_prob = {} #Dictionary that stores log of conditional probability for each token in the training data for each class

N = 0 #Total no. of training docs
for cls in class_dic:
    N += len(training_data[cls])
    
for cls in class_dic:
    Nc = len(training_data[cls]) #No. of docs belonging to class cls
    
    # Calculate prior prob
    if cls not in prior:
        prior[cls] = np.log(Nc/N) #log_e
    
    #Calculate conditonal prob
    if cls not in cond_prob:
        cond_prob[cls] = {}
    
    summ = 0
    for t_ in freq_dic[cls]:
        summ += freq_dic[cls][t_] 
    summ += total_vocab_size #Smoothing (Denominator)
        
    for t in word_dic[cls]:
        tmp = freq_dic[cls][t] + 1 #Smoothing (Numerator)
        cond_prob[cls][t] = np.log(tmp/summ)


# In[12]:

# Save prior, conditional prob as json file 

fp = open(os.path.join(cwd, r'Dataset\prior.json'), 'w+')
json.dump(prior, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\cond_prob.json'), 'w+')
json.dump(cond_prob, fp, sort_keys=True)
fp.close()


# In[13]:

# Load prior, conditional prob

fp = open(os.path.join(cwd, r'Dataset\prior.json'), 'r')
prior = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\cond_prob.json'), 'r')
cond_prob = json.load(fp)
fp.close()


# In[14]:

## NB Testing

acc_count = 0 #Count for correct classification

N_test = 0 #Total no. of test docs
for cls in class_dic:
    N_test += len(test_data[cls])

class_true = [] #List of true classes for each doc in test set
class_found = [] #List of predicted classes for each doc in test set
    
for cls in class_dic:
    test_docs = test_data[cls]
    
    for doc in test_docs:
        
        #Preprocess the doc and form the vocab
        vocab = set()
        
        file_path = count_to_file[doc]
    
        lines = ""
        
        f = open(file_path, 'r')
        lines = f.read()
        
        words = tokenizer.tokenize(lines) #tokenize
        words = [word.strip('_ ').lower() for word in words if word not in STOPWORDS] #lowercase 
        words = [stemmer.stem(word) for word in words] #stemmer
        words = [word for word in words if word not in STOPWORDS and len(word) > 0] #stopword removal

        for word in words:
            vocab.add(word)
            
        #Find score and class
        score = {}
        
        maxx = -float("inf") #Max score for the doc
        maxx_class = cls #Class with the max score
        
        for c in class_dic:
            if c not in score:
                score[c] = 0
            
            score[c] += prior[c]
                
            summ = 0
            for t_ in freq_dic[c]:
                summ += freq_dic[c][t_] 
            summ += total_vocab_size #Smoothing (Denominator)

            for t in vocab:
                if t in cond_prob[c]:
                    score[c] += cond_prob[c][t]
                else: #If the doc contains a term not in the training doc, we use smoothing to avoid -inf
                    tmp = np.log(1/summ)*100
                    score[c] += tmp
                    
            if (score[c] > maxx):
                maxx = score[c]
                maxx_class = c
        
        class_found.append(maxx_class)
        class_true.append(cls)
        
        if (maxx_class == cls): #Predicted class is same as actual class
            acc_count += 1


# In[15]:

# Accuracy
print("Accuracy:", (acc_count/N_test)*100, "%")


# In[16]:

# Confusion Matrix

mat = np.zeros((len(class_dic), len(class_dic)))

for it in range(len(class_found)):
    mat[int(class_true[it])][int(class_found[it])] += 1

df = pd.DataFrame(mat)
print("True vs Predicted")
print(mat)

# Stylish Output
# df = pd.DataFrame(mat)
# cm = sns.light_palette("orange", as_cmap=True)
# df.style.background_gradient(cmap=cm)


# In[17]:

# # Confusion Matirx from Pandas

# from pandas_ml import ConfusionMatrix
# df_confusion = pd.crosstab(pd.Series(class_true), pd.Series(class_found), rownames=['Actual'], colnames=['Predicted'], margins=True)
# df_confusion.style


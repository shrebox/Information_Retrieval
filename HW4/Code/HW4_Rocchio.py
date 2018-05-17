
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
        
fp = open(os.path.join(cwd, r'Dataset\count_to_file.json'), 'w+')
json.dump(count_to_file, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\file_to_count.json'), 'w+')
json.dump(file_to_count, fp, sort_keys=True)
fp.close()


# In[5]:

training_data = {} #Dictionary that stores list of training files for each class
test_data = {} #Dictionary that stores list of test files for each class
class_dic = {} #Dictionary that maps class no to class/folder path
cls_count = 0
ratio = 0.9 #Training Data split ratio

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
    
fp = open(os.path.join(cwd, r'Dataset\training_data.json'), 'w+')
json.dump(training_data, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\test_data.json'), 'w+')
json.dump(test_data, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\class_dic.json'), 'w+')
json.dump(class_dic, fp, sort_keys=True)
fp.close()


# In[6]:

all_words = {} #Unique words in a class
train_tfidf_dic = {} #Inverted index with tfs
train_doc_vector_dic = {} #doc to term mapping with tf-idf (temp)

for cls in class_dic:
    if cls not in all_words:
        all_words[cls] = set() #initialize
        
    files = training_data[cls]
    
    for file in files:
        file_path = count_to_file[file]
        file_id = file_to_count[file_path]
    
        lines = ""
        
        f = open(file_path, 'r')

        local_freq_dic = {} #store only the freq of words appearing in one file

        lines = f.read()

        words = tokenizer.tokenize(lines) #tokenize
        words = [word.strip('_ ').lower() for word in words if word not in STOPWORDS] #lowercase 
        words = [stemmer.stem(word) for word in words] #stemmer
        words = [word for word in words if word not in STOPWORDS and len(word) > 0] #stopword removal

        for word in words:
            all_words[cls].add(word)
            
        for word in words:
            if word not in local_freq_dic:
                local_freq_dic[word] = 0 #initialize
            local_freq_dic[word] += 1

        for word in local_freq_dic:
            if word not in train_tfidf_dic:
                train_tfidf_dic[word] = {} #initialize
            train_tfidf_dic[word][file_id] = local_freq_dic[word]

        if file_id not in train_doc_vector_dic:
            train_doc_vector_dic[file_id] = {}
            for word in local_freq_dic:
                train_doc_vector_dic[file_id][word] = 0 #initialize
             
    all_words[cls] = list(all_words[cls])
    print(len(all_words[cls]))
    
fp = open(os.path.join(cwd, r'Dataset\all_words.json'), 'w+')
json.dump(all_words, fp, sort_keys=True)
fp.close()

for word in train_tfidf_dic:
    df = len(train_tfidf_dic[word].keys())
    idf = (np.log10((train_data_len+1)/df))
    for i in train_tfidf_dic[word]:
        train_tfidf_dic[word][i] = ((1+np.log10(train_tfidf_dic[word][i]))*idf) #Formula for tf-idf
        
for doc_id in train_doc_vector_dic:
    for word in train_doc_vector_dic[doc_id]:
        train_doc_vector_dic[doc_id][word] = train_tfidf_dic[word][doc_id] #Create document vector for tf-idf vector space model
        
fp = open(os.path.join(cwd, r'Dataset\train_tfidf_dic.json'), 'w+')
json.dump(train_tfidf_dic, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\train_doc_vector_dic.json'), 'w+')
json.dump(train_doc_vector_dic, fp, sort_keys=True)
fp.close()


# In[ ]:




# In[7]:

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

fp = open(os.path.join(cwd, r'Dataset\train_tfidf_dic.json'), 'r')
train_tfidf_dic = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\train_doc_vector_dic.json'), 'r')
train_doc_vector_dic = json.load(fp)
fp.close()


# In[8]:

## Rocchio Training

centroid = {}

for cls in class_dic:
    if cls not in centroid:
        centroid[cls] = {}
        for word in all_words[cls]:
            if word not in centroid[cls]:
                centroid[cls][word] = 0
    
    files = training_data[cls]
    
    for file in files:
        for word in all_words[cls]:
            if word in train_doc_vector_dic[file]:
                centroid[cls][word] += train_doc_vector_dic[file][word]
    
    for word in centroid[cls]:
        centroid[cls][word] = (centroid[cls][word]/len(files))  

fp = open(os.path.join(cwd, r'Dataset\centroid.json'), 'w+')
json.dump(centroid, fp, sort_keys=True)
fp.close()

#     print(cls, centroid[cls])


# In[9]:

## Rocchio Testing

test_tfidf_dic = {} #Inverted index with tfs
test_doc_vector_dic = {} #doc to term mapping with tf-idf (temp)

for cls in class_dic:
    files = test_data[cls]
    
    for file in files:
        file_path = count_to_file[file]
        file_id = file_to_count[file_path]
    
        lines = ""
        
        f = open(file_path, 'r')

        local_freq_dic = {} #store only the freq of words appearing in one file

        lines = f.read()

        words = tokenizer.tokenize(lines) #tokenize
        words = [word.strip('_ ').lower() for word in words if word not in STOPWORDS] #lowercase 
        words = [stemmer.stem(word) for word in words] #stemmer
        words = [word for word in words if word not in STOPWORDS and len(word) > 0] #stopword removal

        for word in words:
            if word not in local_freq_dic:
                local_freq_dic[word] = 0 #initialize
            local_freq_dic[word] += 1

        for word in local_freq_dic:
            if word not in test_tfidf_dic:
                test_tfidf_dic[word] = {} #initialize
            test_tfidf_dic[word][file_id] = local_freq_dic[word]

        if file_id not in test_doc_vector_dic:
            test_doc_vector_dic[file_id] = {}
            for word in local_freq_dic:
                test_doc_vector_dic[file_id][word] = 0 #initialize
             
    
for word in test_tfidf_dic:
    if word in train_tfidf_dic:
        df = len(train_tfidf_dic[word].keys()) + 1
    else:
        df = 1
    idf = (np.log10((train_data_len+1)/df))
    for i in test_tfidf_dic[word]:
        test_tfidf_dic[word][i] = ((1+np.log10(test_tfidf_dic[word][i]))*idf) #Formula for tf-idf
        
for doc_id in test_doc_vector_dic:
    for word in test_doc_vector_dic[doc_id]:
        test_doc_vector_dic[doc_id][word] = test_tfidf_dic[word][doc_id] #Create document vector for tf-idf vector space model
        
fp = open(os.path.join(cwd, r'Dataset\test_tfidf_dic.json'), 'w+')
json.dump(test_tfidf_dic, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r'Dataset\test_doc_vector_dic.json'), 'w+')
json.dump(test_doc_vector_dic, fp, sort_keys=True)
fp.close()


# In[10]:

acc_count = 0
class_true = []
class_found = []

for doc in test_doc_vector_dic:
    min_class = -1
    min_valu = 1e9+7
    
    for cls in class_dic:
        centroid_vec = centroid[cls]
        doc_vec = test_doc_vector_dic[doc]
        
        words = set()
        for word in centroid_vec:
            words.add(word)
        for word in doc_vec:
            words.add(word)
            
        words = list(words)
#         print(len(words))
        
        valu = 0
        for word in words:
            if word in centroid_vec and word in doc_vec:
                valu += (centroid_vec[word] - doc_vec[word])**2
            elif word in centroid_vec:
                valu += (centroid_vec[word])**2
            elif word in doc_vec:
                valu += (doc_vec[word])**2
        
        valu = valu**0.5

        #         print(cls, valu)
        if valu < min_valu:
            min_valu = valu
            min_class = cls
    
    class_found.append(min_class)
            
    for file in test_data:
        if doc in test_data[file]:
            actual_class = file
            class_true.append(actual_class)
    
#     print(actual_class, min_class)
    if actual_class == min_class:
        acc_count += 1
        
print("Accuracy:", (acc_count/len(test_doc_vector_dic.keys()))*100, "%")


# In[11]:

mat = np.zeros((len(class_dic), len(class_dic)))

for it in range(len(class_found)):
    mat[int(class_true[it])][int(class_found[it])] += 1

df = pd.DataFrame(mat)
print("True vs Predicted")
print(mat)


import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
import json
import math
import numpy as np
from autocorrect import spell

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('english')
stops = set(stopwords.words('english'))
for i in stops:
    STOPWORDS.add(i)

tf_dic = {} #Inverted index with tfs
count_to_file = {} #File No. to file path mapping
file_to_count = {} #File path to file no. mapping
doc_vector = {} #doc to term mapping with tf-idf (temp)
count = 0 #Total no. of files

cwd = os.getcwd()
folder = os.path.join(cwd, r"stories")

for root, dirs, files in os.walk(folder):
    for file in files:
        if (".html" in file or ".descs" in file or ".header" in file or ".footer" in file or ".musings" in file):
            continue
        
        file_path = os.path.join(root, file)
        count += 1

        lines = ""

        file_to_count[file] = count
        count_to_file[count] = file

        local_freq_dic = {} #store only the freq of words appearing in one file
        
        try:
            f = open(file_path, 'r')
            lines = f.read()

        except Exception as e:
            f = open(file_path, 'r', encoding='mbcs')  #read ansi
            lines = f.read()

        words = tokenizer.tokenize(lines) #tokenize
        words = [word.strip('_ ').lower() for word in words if word not in STOPWORDS] #lowercase 
        words = [stemmer.stem(word) for word in words] #stemmer
        words = [word for word in words if word not in STOPWORDS and len(word) > 0] #stopword removal
        
        for word in words:
            if word not in local_freq_dic:
                local_freq_dic[word] = 0
            local_freq_dic[word] += 1
            
        for word in local_freq_dic:
            if word not in tf_dic:
                tf_dic[word] = []
            tf_dic[word].append([count, local_freq_dic[word]])
            
        if count not in doc_vector:
            doc_vector[count] = []
        for word in local_freq_dic:
            doc_vector[count].append(word)

for word in tf_dic:
    for i in range(len(tf_dic[word])):
        tf_dic[word][i][1] = (1+np.log10(tf_dic[word][i][1]))

f = open(os.path.join(cwd, "file_names.txt"), 'r')

title_dic = {}
while True:
    
    line1 = f.readline()
    if (not line1):
        break
    
    line2 = f.readline()
    
    local_freq_dic = {}

    file_name = line1.split()[0]
    file_title = line2
    
    c = file_to_count[file_name]
    
    words = tokenizer.tokenize(file_title) #tokenize
    words = [word.strip('_ ').lower() for word in words if word not in STOPWORDS] #lowercase 
    words = [stemmer.stem(word) for word in words] #stemmer
    words = [word for word in words if word not in STOPWORDS and len(word) > 0] #stopword removal

    for word in words:
        if (word not in local_freq_dic):
            local_freq_dic[word] = 0
        local_freq_dic[word] += 1

    for word in local_freq_dic:
        if word not in title_dic:
            title_dic[word] = []
        title_dic[word].append([c, local_freq_dic[word]])

for word in title_dic:
    for i in range(len(title_dic[word])):
        title_dic[word][i][1] = (1+np.log10(title_dic[word][i][1]))

for word in title_dic:
    if word not in tf_dic:
        tf_dic[word] = title_dic[word]

for word in title_dic: #title handling
    if word in tf_dic:
        for i in range(len(title_dic[word])):
            for j in range(len(tf_dic[word])):
                if (tf_dic[word][j][0] == title_dic[word][i][0]):
                    tf_dic[word][j][1] = 0.6*(title_dic[word][i][1]) + 0.4*(tf_dic[word][j][1])

    # for word in words:
    #     if word in tf_dic:
    #         res = tf_dic[word]
    #         for doc_tf in tf_dic[word]:
    #             if doc_tf[0] == c:
                    
    #                 for i in res:
    #                     if (i == doc_tf):
    #                         res.remove(i)
                    
    #                 doc_tf[1] += 5 #special attention to title terms
    #                 res.append(doc_tf)
    #         tf_dic[word] = res

idf_dic = {} #idf(t)
wt_dic = {} #inverted index with tf-idf [term to doc]
doc_dic = {} #reversed inverted index with tf-idf [doc to term]

for word in tf_dic:
    idf_dic[word] = np.log10((count+1)/len(tf_dic[word]))
    
for word in tf_dic:
    wt_dic[word] = []
    idf = idf_dic[word]
    
    for doc_tf in tf_dic[word]:
        wt_dic[word].append([doc_tf[0], doc_tf[1] * idf])

for doc_id in doc_vector:
    doc_dic[doc_id] = []
    
    for word in doc_vector[doc_id]:
        for res in wt_dic[word]:
            if (res[0] == doc_id):
                doc_dic[doc_id].append([word, res[1]])

fp = open(os.path.join(cwd, r"tf_dic.json"), 'w+')
json.dump(tf_dic, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r"count_to_file.json"), 'w+')
json.dump(count_to_file, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r"file_to_count.json"), 'w+')
json.dump(file_to_count, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r"idf_dic.json"), 'w+')
json.dump(idf_dic, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r"wt_dic.json"), 'w+')
json.dump(wt_dic, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r"doc_dic.json"), 'w+')
json.dump(doc_dic, fp, sort_keys=True)
fp.close()

fp = open(os.path.join(cwd, r"title_dic.json"), 'w+')
json.dump(title_dic, fp, sort_keys=True)
fp.close()
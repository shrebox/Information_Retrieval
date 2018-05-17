import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
import json
import math
import numpy as np
from autocorrect import spell
from num2words import num2words

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('english')
stops = set(stopwords.words('english'))
for i in stops:
    STOPWORDS.add(i)

cwd = os.getcwd()

fp = open(os.path.join(cwd, r"wt_dic.json"), 'r')
wt_dic = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r"count_to_file.json"), 'r')
count_to_file = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r"file_to_count.json"), 'r')
file_to_count = json.load(fp)
fp.close()

fp = open(os.path.join(cwd, r"doc_dic.json"), 'r')
doc_dic = json.load(fp)
fp.close()

query_cache = []
query_ans_cache = []
query_type = []


while (True):
    print('''Choose one among the following:
            1. Tf-idf based
            2. Cosine similarity based
        Enter Option: ''', end='')
    option = input()

    print("Enter query: ", end='')
    query = input()

    query_terms = tokenizer.tokenize(query) #tokenize
    query_terms = [word.strip('_ ').lower() for word in query_terms if word not in STOPWORDS] #lowercase 

    for word in query_terms: #number correction
        if word.isdigit() == True:
            query_terms.append(num2words(int(word)))

    for i in range(len(query_terms)): #spell checker
        if query_terms[i].isdigit == False:
            query_terms[i] = spell(query_terms[i])

    query_terms = [stemmer.stem(word) for word in query_terms] #stemmer
    query_terms = [word for word in query_terms if word not in STOPWORDS and len(word) > 0] #stopword removal

    final_ans = []

    flag = 0
    for i in range(len(query_cache)):
        if (query_type[i] == option and sorted(query_cache[i]) == sorted(query_terms)):
            print("Cached Output --------")
            for j in query_ans_cache[i]:
                print(j)
            flag = 1

    if (flag == 0):
        if (len(query_cache) == 20):
            del query_cache[0]
            del query_ans_cache[0]
            del query_type[0]
        query_cache.append(query_terms)
        query_type.append(option)

        if (option == '1'):
            relevant_docs = {}

            for word in query_terms:
                if word in wt_dic:
                    for doc_tfidf in wt_dic[word]:
                        doc_id = doc_tfidf[0]
                        wt = doc_tfidf[1]

                        if (doc_id not in relevant_docs):
                            relevant_docs[doc_id] = 0
                        relevant_docs[doc_id] += wt

            if len(relevant_docs) > 0:
                sorted_docs = sorted(relevant_docs, key=relevant_docs.get, reverse=True)
                for doc in sorted_docs[:5]:
                    final_ans.append(count_to_file[str(doc)])

            else:
                final_ans.append("No document found")

        if (option == '2'):
            tf_q = {} #tf dict for query words
            idf_q = {} #idf dict for query words
            wt_q = {} #tf-idf dict for query words
            
            local_freq_dic = {}
            
            for term in query_terms:
                if term not in local_freq_dic:
                    local_freq_dic[term] = 0
                local_freq_dic[term] += 1
            
            for term in local_freq_dic:
                if term not in tf_q:
                    tf_q[term] = []
                tf_q[term].append(local_freq_dic[term])
                
            for term in tf_q:
                idf_q[term] = np.log10((len(doc_dic)+1)/len(tf_q[term]))

            for term in tf_q:
                wt_q[term] = []
                idf = idf_q[term]

                for tf in tf_q[term]:
                    wt_q[term].append((1 + np.log10(tf)) * idf)
                    
        #     print(wt_q)
            
            docs = set() #set of all relevant docs (union)
            for term in wt_q:
                if (term in wt_dic):
                    res = wt_dic[term]
                    for pair in res:
                        docs.add(pair[0])
            
            cos_score = {} #cosine score for each relevant doc
            
            for doc_id in docs:
                #find_cosine_sim(q, doc_id)
                
                doc_id = str(doc_id)
                cos_score[doc_id] = 0
                
                s = 0 #q.d

                for term in wt_q:
                    for pair in doc_dic[doc_id]:
                        if (pair[0] == term):
                            s += (wt_q[term][0]*pair[1]) #compute q.d

                s1 = 0                
                doc_terms = doc_dic[doc_id]
                for i in doc_terms:
                    s1 += (i[1]*i[1]) #compute d^2

                s2 = 0
                for term in wt_q:
                    s2 += (wt_q[term][0]*wt_q[term][0]) #compute q^2

                s1 = math.sqrt(s1) #compute |d|
                s2 = math.sqrt(s2) #compute |q|

                if (s1 == 0 or s2 == 0):
                    s = 0
                else:
                    s = s/(s1*s2) #sim = q.d/|q||d|
            
                cos_score[doc_id] = s
                
            relevant_docs = sorted(cos_score, key=cos_score.get, reverse=True)
            
            if (len(relevant_docs) > 0 and cos_score[relevant_docs[0]] > 0):
                for doc in relevant_docs[:5]:
                    final_ans.append(count_to_file[str(doc)])
            else:
                final_ans.append("No document found")
        
        query_ans_cache.append(final_ans)
        
        for item in final_ans:
            print(item)
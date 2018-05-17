import os
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
ps = PorterStemmer()
import json

folders = os.listdir(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\20_newsgroups")


# print(len(STOPWORDS))
stops = set(stopwords.words('english'))
for i in stops:
    STOPWORDS.add(i)
# STOPWORDS.add(stopwords.words('english'))
# STOPWORDS.union(stops)
# print(len(STOPWORDS))

# words = set()
word_dic = {}
freq_dic = {}
count_to_file = {}
file_to_count = {}
count = 0

for folder in folders:    
    path = os.path.join(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\20_newsgroups", folder)
    
    files_list = os.listdir(path)
    
    for file in files_list:
        count += 1
        path2 = os.path.join(path, file)
        
        file_to_count[path2] = count
        count_to_file[count] = path2
        
        f = open(path2)
        for line in f:
            w = tokenizer.tokenize(line) #tokenizer
            wo = [ps.stem(i) for i in w] #stemmer
            for i in wo:
                i = i.strip()
                i = i.lower()
                if (len(i) > 1 and i not in STOPWORDS): #stop words removal
#                     for punct in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'': #punctation marks removal
#                         i = i.replace(punct, '')
# #                         words.add(i)
                    if (i not in word_dic):
                        word_dic[i] = set()
                        freq_dic[i] = 0
                    word_dic[i].add(count)
                    freq_dic[i] += 1

#         break
#     break
# print(word_dic)
# print(words)
#     print(len(words))
    # print(len(word_dic))

for i in word_dic:
    word_dic[i] = sorted(word_dic[i])
# print(dic['main'])

# k = sorted(word_dic.keys())

# k[80000:80200]

# %matplotlib inline
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',ranks_only=True).generate_from_frequencies(freq_dic)
# plt.figure(figsize=(15,15))
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

wordcloud.to_file(os.path.join(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset", "cloud.png"))

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\data.json", 'w+')
json.dump(word_dic, fp, sort_keys=True)
fp.close()

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\count_to_file.json", 'w+')
json.dump(count_to_file, fp, sort_keys=True)
fp.close()

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\file_to_count.json", 'w+')
json.dump(file_to_count, fp, sort_keys=True)
fp.close()

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\freq_dic.json", 'w+')
json.dump(freq_dic, fp, sort_keys=True)
fp.close()

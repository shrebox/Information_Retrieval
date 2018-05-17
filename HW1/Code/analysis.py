import os
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
ps = PorterStemmer()
import json
import statistics
import math
import matplotlib.pyplot as plt

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\data.json", 'r')
word_dic = json.load(fp)
fp.close()
# print(a['main'])

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\count_to_file.json", 'r')
count_to_file = json.load(fp)
fp.close()

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\file_to_count.json", 'r')
file_to_count = json.load(fp)
fp.close()

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\freq_dic.json", 'r')
freq_dic = json.load(fp)
fp.close()

uniq_freq = set(freq_dic.values())
inv_freq_dic = {}
for i in uniq_freq:
    inv_freq_dic[i] = ''
    
for j in inv_freq_dic:
    for i in freq_dic:
        if(freq_dic[i] == j):
            inv_freq_dic[j] = i
            break

highest = print(max(uniq_freq))	#max
avg = print(statistics.mean(uniq_freq))	#mean
med = print(statistics.median(uniq_freq))	#median
# print(inv_freq_dic[784])
# print(inv_freq_dic['1824'])
# print(len(inv_freq_dic.keys()))
# exit(0)

def xANDy(valx, valy, skips=1):
    res = []
    i = 0
    j = 0
    count_skips = 0
    count_non_skips = 0
    lvalx = len(valx)
    lvaly = len(valy)
    while (i<lvalx and j < lvaly):
        if (valx[i] == valy[j]):
            res.append(valx[i])
            i += 1
            j += 1
        elif (valx[i] < valy[j]):
            if (skips != 1):
                count_skips += 1
                if (i%skips == 0 and i+skips < lvalx and valx[i+skips] < valy[j]):
                    i += skips
                else:
                    i += 1
            else:
                i += 1
                count_non_skips += 1
        else:
            if (skips != 1):
                count_skips += 1 
                if (j%skips == 0 and j+skips < lvaly and valy[j+skips] < valx[i]):
                    j += skips
                else:
                    j += 1
            else:
                j += 1
                count_non_skips += 1
    if (skips == 1):
        print(count_non_skips, end=' ')
    else:
        print(count_skips, end=' ')
    return res

def count_comparisons(string):

	ke = [int(i) for i in inv_freq_dic.keys()]
	ke = sorted(ke)

	for i in ke:
		if (i > 31):
		    print(i, end=' ')
		    xANDy(word_dic[inv_freq_dic[i]], word_dic[string], 1)
		    if (i//2 > 1):
		        xANDy(word_dic[inv_freq_dic[i]], word_dic[string], i//2)
		    if (i//4 > 1):
		        xANDy(word_dic[inv_freq_dic[i]], word_dic[string], i//4)
		    if (i//8 > 1):
		        xANDy(word_dic[inv_freq_dic[i]], word_dic[string], i//8)
		    if (i//16 > 1):
		        xANDy(word_dic[inv_freq_dic[i]], word_dic[string], i//16)
		    if (int(math.sqrt(i)) > 1):
		        xANDy(word_dic[inv_freq_dic[i]], word_dic[string], int(math.sqrt(i)))
		    xANDy(word_dic[inv_freq_dic[i]], word_dic[string], 100)
		    xANDy(word_dic[inv_freq_dic[i]], word_dic[string], 500)
		    xANDy(word_dic[inv_freq_dic[i]], word_dic[string], 1000)
		    xANDy(word_dic[inv_freq_dic[i]], word_dic[string], 10000)
		    print()
		# print(ans1)
		# print(ans2)

def plot_graph(open_file, plot_save_file, axes):
	fp = open(open_file, 'r')

	x1 = []
	y1 = []
	y2 = []
	y3 = []
	y4 = []
	y5 = []
	y6 = []
	y7 = []
	y8 = []
	y9 = []
	y10 = []

	for line in fp:
	    t = line.split()
	    if (int(t[0])%100 == 0):
	        x1.append(t[0])
	        y1.append(t[1])
	        y2.append(t[2])
	        y3.append(t[3])
	        y4.append(t[4])
	        y5.append(t[5])
	        y6.append(t[6])
	        y7.append(t[7])
	        y8.append(t[8])
	        y9.append(t[9])
	#         y10.append(t[10])

	plt.title("Analysis of the Variation in the number of Skips")
	plt.plot(x1, y1, '--o', x1, y2, '.-', x1, y3, '.-', x1, y4, '.-', x1, y5, '.-', x1, y6, 'o-', x1, y7, '.-', x1, y8, '.-', x1, y9, '.-')
	plt.legend(['No skips', 'Skip Size = L/2' , 'L/4', 'L/8', 'L/16', 'sqrt(L)', '100', '500', '1000'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.axis(axes)
	plt.xlabel("Length of Postings List (L) -->")
	plt.ylabel("No. of comparisons -->")
	# plt.figure()
	# plt.show()
	plt.savefig(plot_save_file, bbox_inches="tight")

count_comparisons('edu')	#max
count_comparisons('programm')	#median
count_comparisons('price')	#mean

plot_graph(r"C:/Users/Sanidhya/Documents/IR_HW_1/results_max.txt", r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\analysis_max.png", [0, 1500, 0, 22000])

plt.gcf().clear()

plot_graph(r"C:/Users/Sanidhya/Documents/IR_HW_1/results_mean.txt", r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\analysis_mean.png", [0, 1500, 0, 1900])

plt.gcf().clear()

plot_graph(r"C:/Users/Sanidhya/Documents/IR_HW_1/results_median.txt", r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\analysis_median.png", [0, 1500, 0, 1200])

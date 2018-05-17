import os
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
ps = PorterStemmer()
import json

fp = open(r"C:\Users\Sanidhya\Documents\IR_HW_1\Dataset\data.json", 'r')
a = json.load(fp)
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

print('''Choose one among the following:
        1. x OR y
        2. x AND y
        3. x OR NOT y
        4. x AND NOT y
        5. x AND y (Skip List)
    Enter Option: ''', end='')
option = input()

print("Enter x: ", end='')
x = input()
print("Enter y: ", end='')
y = input()
x.lower()
y.lower()

# option

x = ps.stem(x)
y = ps.stem(y)
print("Root Words:", x, y)
# print(x, y)

# if x in a:
#     print(len(a[x]))
#     print(a[x])
# else:
#     print("Key does not exist")
# if y in a:
#     print(len(a[y]))
#     print(a[y])
# else:
#     print("Key does not exist")

def xORy(valx, valy):
    for item in valy:
        if (item not in valx):
            valx.append(item)
    res = sorted(valx)
    return res

def NOTy(valy):
    all_files = [i for i in range(1, 19998)]
    for i in valy:
        all_files.remove(i)
    return all_files

# def xANDy(valx, valy, skips=100):
#     res = []
#     i = 0
#     j = 0
#     while (i<len(valx) and j<len(valy)):
#         while (i<len(valx) and j<len(valy) and valy[j] < valx[i]):
#             if (skips != 1):
#                 if (j%skips == 0 and j+skips < len(valy) and valy[j+skips] < valx[i]):
#                     j += skips
#                 else:
#                     j += 1
#             else:
#                 j += 1
#         if (i<len(valx) and j<len(valy) and valx[i] == valy[j]):
#             res.append(valy[j])
#         i += 1
#     return res

def xANDy(valx, valy, skips=1):
    res = []
    i = 0
    j = 0
    # count_skips = 0
    # count_non_skips = 0
    lvalx = len(valx)
    lvaly = len(valy)
    while (i<lvalx and j < lvaly):
        if (valx[i] == valy[j]):
            res.append(valx[i])
            i += 1
            j += 1
        elif (valx[i] < valy[j]):
            if (skips != 1):
                # count_skips += 1
                if (i%skips == 0 and i+skips < lvalx and valx[i+skips] < valy[j]):
                    i += skips
                else:
                    i += 1
            else:
                i += 1
                # count_non_skips += 1
        else:
            if (skips != 1):
                # count_skips += 1 
                if (j%skips == 0 and j+skips < lvaly and valy[j+skips] < valx[i]):
                    j += skips
                else:
                    j += 1
            else:
                j += 1
                # count_non_skips += 1
    # if (skips == 1):
    #     print(count_non_skips, end=' ')
    # else:
    #     print(count_skips, end=' ')
    return res

ans = []
valx = []
valy = []
if x in a:
    valx = list(a[x])
if y in a:
    valy = list(a[y])
    
if (option == '1'):
    ans = xORy(valx, valy)
elif (option == '2'):
    ans = xANDy(valx, valy, 1)
elif (option == '3'):
    ans = xORy(valx, NOTy(valy))
elif (option == '4'):
    ans = xANDy(valx, NOTy(valy))
elif (option == '5'):
    print("No. of skips: ", end='')
    skips = int(input())
    ans = xANDy(valx, valy, skips)

print("Found", len(ans), "results !")
# print(ans)
for item in ans:
    print(count_to_file[str(item)])
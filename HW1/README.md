### Assignment 1 (Inverted Index)
**Question**

You need to construct an unigram inverted index on the dataset given at https://github.com/sayhitosandy/Information_Retrieval/tree/master/Dataset/20_newsgroups.zip.

Do complete preprocessing of data: Stop word removal, tokenisation etc. You may use libraries for it but clearly state your assumptions.

Construct a word cloud of the complete dataset.

You need to build a CLI which supports following queries:
1. `x OR y`
2. `x AND y`
3. `x AND NOT y`
4. `x OR NOT y`

Where `x` and `y` would be taken as input from the user.

For query processing, You need to implement
* Merge postings algorithm
* Skip pointers algorithm (only for `x AND y`)

Analyse how the variation in number of skips affects the performance of your system and write in the report.

**Answer**

*Preprocessing:*

(Assumption) Since the removal of header was optional, I have NOT removed header from any file.

For every file in the dataset, we do the following:
1. Extract each line in the file.
2. Tokenize the extracted line using NLTK's tokenizer
3. Apply Porter Stemmer on each token. (For Porter Stemmer, we again use NLTK.)
4. Strip the stemmed/root word of any white spaces and convert it into lower case.
5. Check if the word is a stopword. (The stopwords are chosen from a combination of two libraries - NLTK's stopwords and wordcloud's STOPWORDS. These are 221 in total.)
6. If the word is not a stopword, we remove any punctuations present in it.
7. Then, we add the word to our inverted index.

Our inverted index is a dictionary `word_dic` in which the keys are files numbers from `1` to `19997` and the values are the corresponding document IDs stored in a set, in a sorted order.

In order to map the files numbers to each file, we store another dictionary `count_to_file`. For the reverse mapping, we have a dictionary `file_to_count`. This has been done because the file names are not unique across all the folders. So, we assign unique nonnegative integer IDs to each file (ranging from `1` to `19997`) and store the mapping in the form of the two dictinaries, as mentioned above.

`freq_dic` is another dictionary that stores the frequency of each word in the corpus. This is used to build the word cloud.

The inverted index is stored in the form of a json file, viz, `data.json`. Three more json files are stored for the `freq_dic`, `file_to_count` and `count_to_file` dictionaries.

*Statistics:*

	```Total no. of files: 19997
	Total no. of stopwords: 221
	Vocabulary size (after preprocessing): 184507 words
	```

*Word Cloud:*

I have made use of the libraries `wordcloud` and `matplotlib` to generate the word cloud. It is stored as a `png` file: [cloud.png](https://github.com/sayhitosandy/Information_Retrieval/tree/master/HW1/Results/cloud.png). The word cloud takes the frequencies of each word in the corpus using the `freq_dic`.

*Query Processsing:*

The CLI has 5 options:
1. `x OR y`
2. `x AND y`
3. `x OR NOT y`
4. `x AND NOT y`
5. `x AND y (Skip List)`

We use Porter Stemmer on both `x` and `y` in order to convert them into their root words. For the 1st 4 options, we use the Merge Postings algorithm. For 5th option, we use the Skip Pointers algorithm.

`xORy()`, `NOTy()`, `xANDy()` functions implement these algorithms. For Skip Pointers algorithm, the default skip size is `100`. The user can change this as well.

*Skip Pointers Algorithm Analysis*

See [analysis.pdf](https://github.com/sayhitosandy/Information_Retrieval/blob/master/HW1/Results/analysis.pdf).
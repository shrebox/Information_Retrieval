### Assignment 2 (Document Retrieval)
**Question**

Download http://archives.textfiles.com/stories.zip dataset.

You need to implement a CLI tool for
1. Tf-Idf based document retrieval: For each query, your system will output top `5` documents based on tfidf-matching-score.
2. Tf-Idf based vector space document retrieval: For each query, your system will output top `5` documents based on cosine similarity between query and document vector.

Data preprocessing will be done as similar to the previous assignment. In addition to that, you need to implement [you may use libraries for this]:

* Spelling correction in query
* Consider numerical queries. Example "100 animals", "50,000 variety of flowers", "population of 1 billion", etc.
* Give special attention to the terms in document title

Bonus: Implement query caching [upto last 20 queries]

**Answer**

*Preprocessing:*

Read all the relevant files in the 'stories' folder. There were `477` files in total. Didn't read `.header`, `.footer`, `.html`, `.musings` files as these didn't contain any relevant information. These files were `10` in total.

Also, read the stories in SRE folder. In total, read `467` files.

Some files were encoded in `ansi`. These were caught and read using `ansi` encoding. Rest of the files were read in their default encoding.

Created `4` dictionaries:
1. `wt_dic`: This stored the inverted index. We had a mapping from word to `doc_id` alongwith the `tf-idf` value.
2. `doc_dic`: This dictionary maps `doc_id` to words in that doc alongwith the `tf-idf` value.
3. `count_to_file`: This dictionary maps `doc_id` to file name.
4. `file_to_count`: This dictionary maps file name to `doc_id`.

Stored all these files in the form of `4` json files.

All the preprocessing steps were performed: normalization, tokenization, stopword removal, stemming, etc.

_Special attention to terms in document title:_

For this, created a file `file_names.txt` with the syntax:
	
	<file_name> <file_size>
	<file_title>

Using this file, created a `title_dic` which mapped each word in title to its corresponding `doc_id` alongwith `tf-idf` value.

I also calculated the `td-idf` values of each document using `tf_dic`. This `tf_dic` dictionary mapped each word in the document to its corresponding `doc_id` alongwith `tf-idf` value.

To give higher weight to title words than words in the body of a document, I combined the two dictionaries in the following way: `tf_dic[word].doc->tf-idf = 0.6 * (title_dic[word].doc->tf-idf) + 0.4 * (tf_dic[word].doc->tf-idf)`

*Query Processing:*

The query is processed- normalized, tokenized, stemmed, stopword removed, etc.

The CLI had 2 options:
1. Tf-idf based 
2. Cosine similarity based

For either one, I used the formula taught in the class.

_Spell Check:_

Used library `autocorrect` for this. It has a function `spell()` which does the spelling correction for a given word.

_Number accountance:_

Used library `num2words` for this. It has a function `num2words()` with converts a number into its word form.

So, if a query term is a number, we append its word form in the query itself. This allows us to search for number as well word form of the term.

_Query caching:_

Upto `20` queries can be cached and their results can be stored. So, if a cached query is encountered again, we directly output the result without doing the computations.
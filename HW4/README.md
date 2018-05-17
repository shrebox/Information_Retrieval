### Assignment 4 (Rocchio and KNN)
**Question**

Download `20_newsgroup` dataset from https://github.com/sayhitosandy/Information_Retrieval/tree/master/Dataset/20_newsgroups.zip.

You need to pick documents of `comp.graphics`, `sci.med`, `talk.politics.misc`, `rec.sport.hockey`, `sci.space` [`5`
classes] for text classification.

You need to implement:
1. Rocchio Classification Algorithm
2. KNN classification (vary `k` = `1`, `3`, `5`)

Perform the above steps on `50:50`, `80:20` and `90:10` training and testing split and analyse the accuracy scores.

Compare and Analyse these two methods with previously implemented Naive Bayes Algorithm.

**Answer**

*Preprocessing:*

The data is divided into `70:30` training and test data for each class. Then, vocabulary is made along with all the necessary preprocessing steps. Following are the dictionaries formed:
1. `count_to_file`: Dictionary that maps file no to file path.
2. `file_to_count`: Dictionary that maps file path to file no.
3. `training_data`: Dictionary that stores list of training files for each class.
4. `test_data`: Dictionary that stores list of test files for each class.
5. `class_dic`: Dictionary that maps class no to class/folder path.
6. `train_tfidf_dic`: Dictionary that contains tfidf value for each word in each training document.
7. `test_tfidf_dic`: Dictionary that contains tfidf value for each word in each test document.
8. `train_doc_vector_dic`: Dictionary that contains vector representation of each training document.
9. `test_doc_vector_dic`: Dictionary that contains vector representation of each test document.
10. `centroid`: It stores the centroid for each class. (Only for Rocchio's algo.)

Followed the pseudocode written in the book for both the algorithms.

*Analysis:*

For analysis, we try `4` different splits: `50-50`, `70-30`, `80-20`, `90-10`. We see that with the increase in no. of docs in the training set, the accuracy increases. For more details, please see [analysis.pdf](https://github.com/sayhitosandy/Information_Retrieval/blob/master/HW4/Results/analysis.pdf).
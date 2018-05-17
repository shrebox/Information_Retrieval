CSE508 Information Retrieval Assignment 4

Dataset: https://github.com/sayhitosandy/Information_Retrieval/Dataset/20_newsgroups.zip

Preprocessing:
	The data is divided into 70:30 training and test data for each class. Then, vocabulary is made along with all the necessary preprocessing steps. Following are the dictionaries formed:
	1.	count_to_file: Dictionary that maps file no to file path
	2.	file_to_count: Dictionary that maps file path to file no
	3.	training_data: Dictionary that stores list of training files for each class
	4.	test_data: Dictionary that stores list of test files for each class
	5.	class_dic: Dictionary that maps class no to class/folder path
	6.	train_tfidf_dic: Dictionary that contains tfidf value for each word in each training document.
	7.	test_tfidf_dic: Dictionary that contains tfidf value for each word in each test document.
	8.	train_doc_vector_dic: Dictionary that contains vector representation of each training document.
	9.	test_doc_vector_dic: Dictionary that contains vector representation of each test document.
	10:	centroid: It stores the centroid for each class. (Only for Rocchio's algo.)

Followed the pseudocode written in the book for both the algorithms.

Analysis:
	For analysis, we try 4 different splits: 50-50, 70-30, 80-20, 90-10. We see that with the increase in no. of docs in the training set, the accuracy increases. For more details, please see analysis.pdf.

Sanidhya Singal 2015085
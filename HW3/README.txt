CSE508 Information Retrieval Assignment 3

Dataset: https://github.com/sayhitosandy/Information_Retrieval/Dataset/20_newsgroups.zip

Preprocessing:
	The data is divided into 70:30 training and test data for each class. Then, vocabulary is made along with all the necessary preprocessing steps. Following are the dictionaries formed:
	1.	count_to_file: Dictionary that maps file no to file path
	2.	file_to_count: Dictionary that maps file path to file no
	3.	training_data: Dictionary that stores list of training files for each class
	4.	test_data: Dictionary that stores list of test files for each class
	5.	class_dic: Dictionary that maps class no to class/folder path
	6.	word_dic: Dictionary that stores vocabulary for each class
	7.	freq_dic: Dictionary that stores the frequency of occurance of each word in the vocabulary for each class

NB Training:
	Followed the NB Training algorithm written in the book. Also performed smoothing (Add 1 smoothing).
	Following are the dictionaries made:
	1.	prior: Dictionary that stores log of prior probability for each class
	2.	cond_prob: Dictionary that stores log of conditional probability for each token in the training data for each class

NB Testing:
	Followed the NB Testing algorithm written in the book. For words present in test data and not present in training data, i.e., the unknown words, we again perform smoothing with term freq as zero and then, penalising it 100 times.
	We output accuracy and confusion matrix.

Analysis:
	For analysis, we try 4 different splits: 50-50, 70-30, 80-20, 90-10. We see that with the increase in no. of docs in the training set, the accuracy increases. For more details, please see analysis.pdf.

Sanidhya Singal 2015085
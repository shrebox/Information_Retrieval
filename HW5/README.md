### Assignment 5 (K-Means Clustering / Network Analysis)
**Question 1**

Download `20_newsgroup` dataset from https://github.com/sayhitosandy/Information_Retrieval/tree/master/Dataset/20_newsgroups.zip.

You need to pick documents of `comp.graphics`, `sci.med`, `talk.politics.misc`, `rec.sport.hockey`, `sci.space` [`5` classes] for text classification.

You need to use the below as feature vectors:
1. Bag of Words Model
2. Word2Vec representation from Google News Pretrained Word2Vec model [you can refer to: http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/ ]

For both of these features set, implement K-means clustering algorithm [you cannot use any library for
k-means] [don’t use groundtruth information] and report the error.

Draw your inferences.

**Question 2**

Choose any real world network [e.g. from https://snap.stanford.edu/data/index.html ]

Describe your network briefly in terms of nodes, edges, etc.

Make sure to choose a network of less than 1000 nodes or randomly subsample nodes from available data. You need to [don’t use any library for any of these tasks]:
* Plot degree distribution of the network
* Calculate clustering coefficient for each node
* Calculate betweenness and closeness centrality for each node

What can you infer about the network. State your observations.

**Answer 2**

*Real World Network:*

http://networkrepository.com/socfb-Haverford76.php

It contains `1.4K` nodes and `59.6K` edges. A social friendship network extracted from Facebook consisting of people (nodes) with edges representing friendship ties.

The main file is in the form of `mtx (Matrix Market)` file. It is extracted to Adjacency Matrix using `scipy`.

*Tasks:*

All the tasks are performed directly on the graph. The outputs are stored as `png` figures and `json` files. Read [analysis.pdf](https://github.com/sayhitosandy/Information_Retrieval/blob/master/HW5/Results/analysis.pdf) for details.

*Formulae:*

1. **Degree Distribution**
	
	[Referred from Wikipedia]
	```
	P(k) = n_k/n
	where 	n_k = No of nodes with degree k
			n = Total no of nodes
	```

2. **Clustering Coefficient**
	
	[Referred from https://www.youtube.com/watch?v=K2WF4pT5pFY ]
	```
	CC(v) = 2 * Nv / (kv * (kv - 1))
	where	Nv = No of links between the neighbours of v
			kv = Degree of v
	```

3. **Closeness Centrality**
	
	Ran BFS on all nodes, computed distances and took mean. [Referred from slides]

4. **Betweenness Centrality**
	
	[Referred from http://www.cl.cam.ac.uk/teaching/1617/MLRD/handbook/brandes.pdf ]
	
	Implemented Brandes algorithm.
CSE508 Information Retrieval Assignment 5

Real World Network:
	http://networkrepository.com/socfb-Haverford76.php
	It contains 1.4K nodes and 59.6K edges. A social friendship network extracted from Facebook consisting of people (nodes) with edges representing friendship ties.

	The main file is in the form of mtx (Matrix Market) file. It is extracted to Adjacency Matrix using scipy.

Tasks:
	All the tasks are performed directly on the graph. The outputs are stored as png figures and json files. Read analysis.pdf for details.

Formulae:
	1.	Degree Distribution
			Referred from Wikipedia
			P(k) = n_k/n
			where 	n_k = No of nodes with degree k
					n = Total no of nodes

	2.	Clustering Coefficient
			Referred from https://www.youtube.com/watch?v=K2WF4pT5pFY
			CC(v) = 2 * Nv / (kv * (kv - 1))
			where	Nv = No of links between the neighbours of v
					kv = Degree of v

	3.	Closeness Centrality
			Ran BFS on all nodes, computed distances and took mean. Referred from slides.

	4.	Betweenness Centrality
			Referred from http://www.cl.cam.ac.uk/teaching/1617/MLRD/handbook/brandes.pdf.
			Implemented Brandes algorithm.

Sanidhya Singal 2015085
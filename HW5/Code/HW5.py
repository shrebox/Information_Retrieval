
# coding: utf-8

# In[1]:

import scipy.io
import matplotlib.pyplot as plt
import os
import json
import queue


# In[2]:

a = scipy.io.mmread('Networks/socfb-Haverford76.mtx')


# In[3]:

b = a.toarray()
# print(b.shape)


# In[4]:

nodes = set()
adjlist = {}

for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        if (b[i][j] == 1):
            nodes.add(i)
nodes = list(nodes)

for i in nodes:
    if i not in adjlist:
        adjlist[i] = []
        
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        if (b[i][j] == 1):
            adjlist[i].append(j)

for i in adjlist:
    adjlist[i] = sorted(adjlist[i])


# In[5]:

# print(len(nodes))


# In[6]:

# print(adjlist[0])


# In[7]:

# Degree distribution
degree = {}
for i in nodes:
    if i not in degree:
        degree[i] = 0
        
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        if (b[i][j] == 1):
            degree[i] += 1


# In[8]:

maxx = max(degree.values())


# In[9]:

deg_dist = {}
for i in range(0, maxx+1):
    if (i not in deg_dist):
        deg_dist[i] = 0
        
for i in degree:
    deg_dist[degree[i]] += 1
    
for i in deg_dist:
    deg_dist[i] = deg_dist[i]/len(nodes)

fp = open(os.path.join(os.getcwd(), r'Degree_Distribution.json'), 'w+')
json.dump(deg_dist, fp, sort_keys=True)
fp.close()


# In[10]:

x = [i for i in deg_dist]
y = [deg_dist[i] for i in deg_dist]
plt.plot(x,y,'b-',x,y,'r.')
plt.xlabel('Degree k')
plt.ylabel('P(k) = nk/n')
plt.title('Degree Distribution')
plt.grid(True)
plt.savefig('Degree_Distribution.png')
plt.clf()


# In[11]:

# Clustering Coefficient
clus_coeff = {}

for i in nodes:
    nv = 0
    ngbr = adjlist[i]
    for j in range(0, len(ngbr)-1):
        for k in range(j+1, len(ngbr)):
            if (ngbr[k] in adjlist[ngbr[j]]):
                nv+=1
    if (degree[i] > 1):
        cc = (2*nv)/(degree[i] * (degree[i] - 1))
        clus_coeff[i] = cc
    else:
        clus_coeff[i] = 0


# In[12]:

fp = open(os.path.join(os.getcwd(), r'Clustering Coeff.json'), 'w+')
json.dump(clus_coeff, fp, sort_keys=True)
fp.close()


# In[13]:

x = [i for i in clus_coeff]
y = [clus_coeff[i] for i in clus_coeff]
plt.plot(x,y,'b-',x,y,'r.')
plt.xlabel('Nodes')
plt.ylabel('Clustering Coefficient')
plt.title('Clustering Coefficient Distribution')
plt.grid(True)
plt.savefig('Clustering_Coefficient.png')
plt.clf()


# In[14]:

# Closeness Centrality


# In[15]:

def closeness_centrality(start): #Basic BFS
    dist = {}
    for i in nodes:
        if i not in dist:
            dist[i] = 999999

    dist[start] = 0
    q = queue.Queue()
    q.put(start)
    
    while (not q.empty()):

        n = q.get()
        ngbrs = adjlist[n]
        
        for ngbr in ngbrs:
            if (dist[n] + 1 < dist[ngbr]):
                dist[ngbr] = dist[n] + 1
                q.put(ngbr)

    s = 0
    for i in dist.values():
        s += i
    return (s/len(dist))


# In[16]:

closeness_cent = {}
for i in nodes:
    if i not in closeness_cent:
        closeness_cent[i] = 0

for i in nodes:
    closeness_cent[i] = closeness_centrality(i)
    
fp = open(os.path.join(os.getcwd(), r'Closeness Centrality.json'), 'w+')
json.dump(closeness_cent, fp, sort_keys=True)
fp.close()


# In[17]:

x = [i for i in closeness_cent]
y = [closeness_cent[i] for i in closeness_cent]
plt.plot(x,y,'b-',x,y,'r.')
plt.xlabel('Nodes')
plt.ylabel('Closeness Centrality')
plt.title('Closeness Centrality Distribution')
plt.grid(True)
plt.savefig('Closeness_Centrality.png')
plt.clf()


# In[18]:

# Betweeness Centrality (Brandes Algo) ... http://www.cl.cam.ac.uk/teaching/1617/MLRD/handbook/brandes.pdf

def betweeness_centrality():
    bet_cent = {}
    for i in nodes:
        if i not in bet_cent:
            bet_cent[i] = 0

    for start in nodes:
        stack = []

        pred = {}
        for i in nodes:
            if i not in pred:
                pred[i] = []

        paths = {}
        for i in nodes:
            if i not in paths:
                paths[i] = 0

        dist = {}
        for i in nodes:
            if i not in dist:
                dist[i] = -1

        paths[start] = 1
        dist[start] = 0

        q = queue.Queue()
        q.put(start)
        distance=0

        while (not q.empty()):
            n = q.get()
            stack.append(n)
            ngbrs = adjlist[n]

            for ngbr in ngbrs:
                if dist[ngbr] < 0:
                	nlen = dist[ngbr]
                	dist[ngbr] = dist[n] + 1
                	distance += nlen
                	q.put(ngbr)

                if dist[ngbr] == dist[n] + 1:
                    pred[ngbr].append(n)
                    paths[ngbr] += paths[n]

        delta = {}
        for i in nodes:
            if i not in delta:
                delta[i] = 0

        while stack:
            node = stack.pop()

            parents = pred[node]
            for parent in parents:
                delta[parent] += ((paths[parent]/paths[node]) * (1 + delta[node]))

                if node != start:
                    bet_cent[node] += delta[node]

    return bet_cent


# In[19]:

between_cent = betweeness_centrality()

max_bc = max(between_cent.values())
min_bc = min(between_cent.values())

for i in between_cent:
    between_cent[i] = (between_cent[i] - min_bc)/(max_bc - min_bc) #normalize (Wikipedia)

fp = open(os.path.join(os.getcwd(), r'Between Centrality.json'), 'w+')
json.dump(between_cent, fp, sort_keys=True)
fp.close()


# In[20]:

x = [i for i in between_cent]
y = [between_cent[i] for i in between_cent]
plt.plot(x,y,'b-',x,y,'r.')
plt.xlabel('Nodes')
plt.ylabel('Betweenness Centrality')
plt.title('Betweenness Centrality Distribution')
plt.grid(True)
plt.savefig('Betweenness_Centrality.png')
plt.clf()


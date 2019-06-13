from scipy.sparse import *
from sklearn.cluster import KMeans

from scipy import *
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import math

train = []
with open('train.dat') as f:
    for line in f:
        train.append(line)


#making the matrix from data

def makematrix(train):
    index = []
    val = []
    ptr = []
    ptrprev = 0
    ptr.append(ptrprev)
    for lines in train:
        o = lines.split(" ")
        ptrprev += (len(o) // 2)
        ptr.append(ptrprev)
        for iters, nums in enumerate(o):
            if iters % 2 == 0:
                index.append(int(nums))
            else:
                val.append(int(nums))

    return index, val, ptr


ind, val, ptr = makematrix(train)

nrows = len(train)
ncol = len(ind)



#making it a csr matrix
csrmat=csr_matrix( (val,ind,ptr),shape=(nrows, ncol),dtype=np.double )

csrmat.sort_indices()

##Dimensionality Reduction

dim = TruncatedSVD(n_components=200)
mat = dim.fit_transform(csrmat)
del csrmat
mat = csr_matrix(mat)


# TAKEN FROM INCLASS ACTIVITY
def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm.
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i + 1]):
            rsum += val[j] ** 2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0 / np.sqrt(rsum)
        for j in range(ptr[i], ptr[i + 1]):
            val[j] *= rsum

    if copy is True:
        return mat


mat_normalized = csr_l2normalize(mat, copy=True)

kmeans = KMeans(n_clusters=150, random_state=0).fit(mat_normalized)
#centroid=kmeans.cluster_centers_
labels=kmeans.labels_


def dbscan(mat, labels, minpts, eps):
    core = [np.nan] * len(train)
    label = [np.nan] * len(train)
    visited = [np.nan] * len(train)
    noise = [np.nan] * len(train)
    cluster = 1
    for i in range(150):
        for iters, val in enumerate(mat):
            if labels[iters] == i and not math.isnan(visited[iters]):
                # border point
                # if not math.isnan(core[iters]):
                continue
            elif labels[iters] == i and math.isnan(visited[iters]):
                n = []
                for iters1, v in enumerate(mat):

                    if labels[iters1] == i and iters1 == iters:
                        continue
                    elif labels[iters1] == i:
                        sim = cosine_similarity(v, mat[iters])
                        if sim[0][0] > eps:
                            n.append(iters1)
                if len(n) > minpts:
                    core[iters] = 1
                    label[iters] = cluster
                    visited[iters] = 1
                    for neigh in n:
                        if math.isnan(visited[neigh]):
                            label[neigh] = cluster
                            visited[neigh] = 1
                        else:
                            # label[neigh]=cluster #check, because if it is already visited, do not reassign
                            # border[neigh]=1
                            visited[neigh] = 1
                # elif len(n)>0:
                # border[iters]=1
                # visited[iters]=1
                else:
                    noise[iters] = 1
        cluster += 1
        #print(label)
    return label, noise,cluster



l,n,c=dbscan(mat_normalized,labels,19,0.25)

#print(c)
for iters,values in enumerate(l):
    if math.isnan(values):
        l[iters]=c+1
f = open('clusterlabelsminpts190.25.txt', 'w+')
for item in l:
    f.write("%s\r\n" % (item))
f.close()

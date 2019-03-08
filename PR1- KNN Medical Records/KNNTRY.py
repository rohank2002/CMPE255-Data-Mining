import nltk
import numpy as np
import re
import string
from nltk.stem.lancaster import LancasterStemmer
import math
from scipy.sparse import csr_matrix
#from statistics import mode


nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords #stack overflow
from nltk.tokenize import word_tokenize #geeks for geeks


#importing train data from file
cls = []
data = []
with open('trainf.dat') as file:
    content = file.readlines()
    for lines in content:
        lines = lines.strip('\n')
        lines = lines.strip('\t')
        cls.append(lines[0])
        data.append(lines[2:])
#print(data[0])

# importing test data
pred = []

with open('testf.dat') as file:
    content = file.readlines()
    for lines in content:
        pred.append(lines)
#print(pred[0])

#Preprocessing the data (removing stop words )
def preprocess(data):
    output = []
    i = 0
    for entry in data:
        output = re.sub(r'\d+', '', entry)
        data[i] = output
        i += 1

#removing prepositions and unecessary words
stop_words = set(stopwords.words('english'))
i = 0
for profiles in data:
        word_tokens = word_tokenize(profiles.lower())
        output = ''
        for w in word_tokens:
            if w not in stop_words:
                output = output + ' ' + w
        data[i] = output
        i += 1
#removing punctuations and symbols

i = 0
for profiles in data:
    data[i] = profiles.translate(str.maketrans('','',string.punctuation)) #stackover flow
    i += 1
#stemming
i = 0
for profiles in data:
    ps = LancasterStemmer()
    tokens = word_tokenize(profiles)
    output = ''
    for w in tokens:
        w = ps.stem(w)
        output = output + ' ' + w
    data[i] = output
    i += 1
preprocess(data)

#Splitting the data as list of lists
splittrain=len(data)

docs = []
for i in range(0, splittrain):
    docs.append(data[i].split())
# Creating the csr matrix

indptr = [0]
indices = []
traindata = []
vocabulary = {}

for profiles in docs:
    for  words in profiles:
        index = vocabulary.setdefault(words, len(vocabulary))
        indices.append(index)
        traindata.append(1)
    indptr.append(len(indices))
trainmatrix = csr_matrix((traindata, indices, indptr), dtype=float).toarray()
length = len(trainmatrix)
#print(length)


j = 0
for entry in trainmatrix:
    lengthdoc = len(docs[j])
    sumcol = 0
    for i in range(0, length):
        if trainmatrix[i,j] != 0: sumcol += 1
    idf = 1 + math.log(length/sumcol, 10)
    trainmatrix[:,j] *= idf
    trainmatrix[j,:] /= lengthdoc
    j += 1

#preprocess test
preprocess(pred)
lengpred = len(pred)

testdocs = []
for i in range(0, lengpred):
    testdocs.append(pred[i].split())


indptr2 = [0]
indices2 = []
testdata = []

#CSR for test data

for profiles in testdocs:
    for words in profiles:
        if words in vocabulary:
            index = vocabulary.setdefault(words, len(vocabulary))
            indices2.append(index)
            testdata.append(1)
    indptr2.append(len(indices2))
testmatrix = csr_matrix((testdata, indices2, indptr2), shape = (lengpred, len(vocabulary)), dtype=float).toarray()
length = len(testmatrix)


j = 0
for elements in testmatrix:
    lengthdoc = len(testdocs[j])
    sumc = 0
    for i in range(0, length):
        if testmatrix[i,j] != 0: sumc += 1
    if sumc != 0:
        idf = 1 + math.log(lengpred/sumc, 10)
    testmatrix[:,j] *= idf
    testmatrix[j,:] /= lengthdoc
    j += 1

#Calculating Cosine similarity

testvector = [0]*lengpred
trainvector = [0]*splittrain
i = 0
x = []
x = testmatrix**2
sm = x.sum(axis=1, dtype=float) # sum the rows
testvector = np.sqrt(sm)

y = []
y = trainmatrix**2
sm = y.sum(axis=1, dtype=float)
trainvector = np.sqrt(sm)


final = [[0] * splittrain for i in range(lengpred)]
#print(final)
test = 0
train = 0
x = []
for i in range(0, lengpred):
    dots = []
    x = testmatrix[i]
    dots = trainmatrix.dot(x)
    test = testvector[i]
    for k in range(0, splittrain):
        final[i][k] = (dots[k]/(test*trainvector[k]))
#print(len(final[0]))
#c=final
#c.sort(reverse=True)
#print(c)

#print(c[1])
#print(op)

#get Neighbours

def kneighbours(final, k, cls):
    op=[]
    for i in range(0, len(final)):
        sim = []
        for j in range(0, k):
            ind = np.argmax(final[i])  #get maximum vote
            sim.append(ind)
            final[i][ind] = 0
            #print(sim)
        clsfinal = []
        for k in range(0, len(sim)):
            clsfinal.append(int(cls[sim[k]]))

            op.append(max(clsfinal, key = clsfinal.count))
    return op


op=kneighbours(final, 10, cls)
f=open('op.txt','w+')
for item in op:
    f.write("%d\r\n"%(int(item)) )
f.close()















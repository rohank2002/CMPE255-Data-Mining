

import glob
import cv2
import pandas as pd
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier

print("importing train")
train = []
images = {}
train_names=[]
for imagePath in glob.glob("/home/013759252/traffic/train"+"/*.jpg"):
    name=imagePath.split("/")[-1]
    train_names.append(name)
    image = cv2.imread(imagePath)
    
    resized = cv2.resize(image, (64, 64))
    gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hist= feature.hog(gray_scale, orientations=10, pixels_per_cell=(10, 10),cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1")
    train.append(hist)
traindf=pd.DataFrame()
traindf["name"]=train_names
traindf["data"]=train
traindf=traindf.sort_values(["name"])
#print(traindf)
print("importing test")

test_names=[]
test=[]
for imagePath in glob.glob("/home/013759252/traffic/test"+"/*.jpg"):
    name=imagePath.split("/")[-1]
    test_names.append(name)
    image = cv2.imread(imagePath)
    resized = cv2.resize(image, (64, 64))
    gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hist= feature.hog(gray_scale, orientations=10, pixels_per_cell=(10, 10),cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1")

    test.append(hist)
testdf=pd.DataFrame()
testdf["name"]=test_names
testdf["data"]=test
testdf=testdf.sort_values(["name"])

#print(traindf)
trainl=traindf["data"].tolist()
testl=testdf["data"].tolist()



labels = []
print("importing labels")
with open('/home/013759252/traffic/train.labels') as file:
    content = file.readlines()

    for lines in content:
        lines = lines.strip('\n')
        labels.append(lines)
# print(len(labels))
#print(trainl[0])
print("starting to model")

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(trainl, labels)

predictions = neigh.predict(testl)

print("printing to file")


f = open('ophog.txt', 'w+')
for item in predictions:
    f.write("%s\r\n" % (item))
f.close()
print("DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")








import torch
import numpy as np
from os import listdir
from os.path import isfile, join
import sys

mypath = "/media/minh/New Volume/LINUX/jpeg/train"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.pt' in f]

distanceMatrix = []
indexMatrix = []

# preload the neighborTensor
neighborTensor = np.empty((1,2048))
for neighborFile in onlyfiles:
    print("Loading: " + neighborFile)
    neighborVector = torch.load(mypath + "/" + neighborFile).detach().numpy()
    np.append(neighborTensor, neighborVector, axis = 0)
print("Done Preloading!")

for thisFile in onlyfiles:
    print("Making distance vector for " + thisFile)
    distanceVector = []
    thisVector = torch.load(mypath + "/" + thisFile).detach().numpy()
    thisTensor = np.repeat(thisVector, len(onlyfiles), axis=0)
#------------------------------- parallel code ---------------------------------
    # The following loop is parallelized and is kept to show what logically happens

    # for neighborFile in onlyfiles:
    #     if (thisFile == neighborFile):
    #         continue
    #     else:
    #         neighborVector = torch.load(mypath + "/" + neighborFile).detach().numpy()
    #         dist = np.linalg.norm(neighborVector - thisVector)
    #         distanceVector.append(dist)
    # sortedDistanceVector = sorted(distanceVector, key = lambda x:float(x))
    # sortedDistanceIndex  = sorted(range(len(distanceVector)), key = lambda x:distanceVector[x])

    diff = np.subtract(thisTensor, neighborTensor)
    distanceVector = np.linalg.norm(diff,axis=0)
    sortedDistanceVector = np.sort(distanceVector, axis=0)
    sortedDistanceIndex = np.argsort(distanceVector,axis=0)
#-------------------------------end parallel code ------------------------------
    print(sortedDistanceVector)
    print(sortedDistanceIndex)
    distanceMatrix.append(sortedDistanceVector)
    indexMatrix.append(sortedDistanceIndex)
    break
distanceMatrix = torch.FloatTensor(distanceMatrix)
torch.save(distanceMatrix,mypath + "/distanceMatrix")

indexMatrix = torch.FloatTensor(indexMatrix)
torch.save(indexMatrix,mypath + "/indexMatrix")

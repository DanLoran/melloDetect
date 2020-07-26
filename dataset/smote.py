import torch
import numpy as np
from os import listdir
from os.path import isfile, join

mypath = "/media/minh/New Volume/LINUX/jpeg/train"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and '.pt' in f]

distanceMatrix = []
indexMatrix = []
for thisFile in onlyfiles:
    print("Making distance vector for " + thisFile)
    distanceVector = []
    thisVector = torch.load(mypath + "/" + thisFile).detach().numpy()
    for neighborFile in onlyfiles:
        #print("\t Checking with neighbor: " + neighborFile)
        if (thisFile == neighborFile):
            continue
        else:
            neighborVector = torch.load(mypath + "/" + neighborFile).detach().numpy()
            dist = np.linalg.norm(neighborVector - thisVector)
            distanceVector.append(dist)
    #assert(len(distanceVector) == len(onlyfiles) - 1)
    sortedDistanceVector = sorted(distanceVector, key = lambda x:float(x))
    sortedDistanceIndex  = sorted(range(len(distanceVector)), key = lambda x:distanceVector[x])
    print(sortedDistanceVector[0:4])
    print(sortedDistanceIndex[0:4])
    distanceMatrix.append(sortedDistanceVector)
    indexMatrix.append(sortedDistanceIndex)

distanceMatrix = torch.FloatTensor(distanceMatrix)
torch.save(distanceMatrix,mypath + "/distanceMatrix")

indexMatrix = torch.FloatTensor(indexMatrix)
torch.save(indexMatrix,mypath + "/indexMatrix")

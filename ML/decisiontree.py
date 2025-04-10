import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshhold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshhold = threshhold
        self.left = left
        self.right = right
        self.value = value
        #print("Node initiated")


    def isLeafNode(self):
        return self.value is not None
    

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, nrFeatures=None):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.nrFeatures = nrFeatures
        self.root = None
        #print("Decision Tree initiated")


    def fit(self, X, y):
        #print("Fitting")
        if self.nrFeatures is None:
            self.nrFeatures = int(np.sqrt(X.shape[1]))
        self.root = self._buildTree(X, y)


    def _buildTree(self, X, y, depth=0):
        nSamples, nFeatures = X.shape
        nLabels = len(np.unique(y))
        
        #Check for stop
        if (depth>=self.max_depth or nLabels ==1 or nSamples<self.min_samples_split):
            leafValue = self._mostCommonLabel(y)
            #print("Leaf Node found")
            return Node(value=leafValue)
        

        featIndex = np.random.choice(nFeatures, self.nrFeatures, replace=False)
        bestFeature, bestThresh = self._bestSplit(X, y, featIndex)
        
        lIdxs, rIndxs = self._split(X[:, bestFeature], bestThresh)

        if len(lIdxs) == 0 or len(rIndxs) == 0:
            leafValue = self._mostCommonLabel(y)
            return Node(value=leafValue)
        
        left = self._buildTree(X[lIdxs, :], y[lIdxs], depth+1)
        right = self._buildTree(X[rIndxs, :], y[rIndxs], depth+1)

        return Node(bestFeature, bestThresh, left, right)



    
    def _mostCommonLabel(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    

    def _bestSplit(self, X, y, featIndex):
        bestGain = -1
        splitIndex, splitThreshhold = None, None
 
        for fIndx in featIndex:
            X_column = X[:, fIndx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._infoGain(y,X_column,threshold)
                if gain > bestGain:
                    bestGain = gain
                    splitIndex = fIndx
                    splitThreshhold = threshold
            
        return splitIndex, splitThreshhold

    def _infoGain(self, y, X_column, threshold):
        parentEntropy = self._entropy(y)

        leftIdx, rightIdx = self._split(X_column, threshold)

        if len(leftIdx) == 0 or len(rightIdx) == 0:
            return 0

        n = len(y)
        nLeft, nRight = len(leftIdx), len(rightIdx)
        entLeft, entRight = self._entropy(y[leftIdx]), self._entropy(y[rightIdx])
        childEntropy = (nLeft/n) * entLeft + (nRight/n) * entRight

        infoGain = parentEntropy - childEntropy
        return infoGain


    def _split(self, xCol, splitThresh):
        leftIdxs = np.argwhere(xCol<=splitThresh).flatten()
        rightIdxs = np.argwhere(xCol>splitThresh).flatten()
        return leftIdxs, rightIdxs


    #Calc entropy
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def predict(self, X):
        return np.array([self._travTree(x, self.root) for x in X])


    def _travTree(self, x, node):
        if node.isLeafNode():
            return node.value
            
        if x[node.feature] <= node.threshhold:
            return self._travTree(x, node.left)
        return self._travTree(x, node.right)
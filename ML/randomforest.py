from decisiontree import DecisionTree
import numpy as np
from collections import Counter
import logging


logger = logging.getLogger(__name__)
class RandomForest:

    #Initiate forest algo
    def __init__(self, nrTrees=50, maxDepth = 5, minSampleSplit=2, nrFeatures = None):
        self.nrTrees = nrTrees
        self.maxDepth = maxDepth
        self.minSampleSplit = minSampleSplit
        self.nrFeatures = nrFeatures
        self.trees = []
        logger.info("Random Forest initialized")
        logger.info(f"nrTrees: {nrTrees}")
        logger.info(f"maxDepth: {maxDepth}")
        logger.info(f"minSampleSplit: {minSampleSplit}")

    #Create forest
    def fit(self, X, y):
        if self.nrFeatures is None:
            self.nrFeatures = int(np.sqrt(X.shape[1]))

        self.trees = []
        for n in range(self.nrTrees):
            tree = DecisionTree(max_depth=self.maxDepth,
                         min_samples_split=self.minSampleSplit,
                         nrFeatures=self.nrFeatures)
            xSample, ySample = self._bootstrapSamples(X, y)
            tree.fit(xSample, ySample)
            self.trees.append(tree)
            


    def _bootstrapSamples(self, X, y):
        nrSamples = X.shape[0]
        indexes = np.random.choice(nrSamples, nrSamples, replace=True)
        return X[indexes], y[indexes]



    def predict(self, X):
        predictions = np.array([tree.predict(X)for tree in self.trees])
        treePredictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._mostCommonLabel(pred) for pred in treePredictions])
        return predictions

    def _mostCommonLabel(self, y):
        counter = Counter(y)
        commonValue = counter.most_common(1)[0][0]
        return commonValue


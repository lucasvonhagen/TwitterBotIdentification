import numpy as np
from ML.randomforest import RandomForest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ML.preproc import preprocess_tweet_data
from ML.preproc import combine_data
import logging


#TEST
#data = load_breast_cancer()
#X, y = data.data, data.target
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Load data
combined_data = combine_data()
features, target = preprocess_tweet_data(combined_data)

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Evaluate accuracy
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def acc(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)


#Global logging configuration
logging.basicConfig(
    filename='stats.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Evaluate accuracy
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def acc(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

def run_forest(nrTrees, maxDepth):
    logger.info("RandomForest started")
    forest = RandomForest(nrTrees, maxDepth)
    forest.fit(X_train, y_train)
    predictions = forest.predict(X_test)
    acc_value = acc(y_test, predictions)
    logger.info(f"Accuracy: {acc_value}")
    print("Accuracy: ", acc_value)
    logger.info("RandomForest Ended")

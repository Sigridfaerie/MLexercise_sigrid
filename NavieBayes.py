#this is the answer of question 3
from collections import Counter

#abtain data_set
from math import exp,pi,sqrt
import numpy as np
from numpy import ndarray, exp, pi, sqrt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

iris= load_iris()
X, y = iris['data'], iris['target']
print(X,y)
#shuffle the data_set at the same time(keep the x,y are corresponding still
#print(y)
N, D = X.shape
#print(N,D)
Ntrain = int(0.8 * N)
shuffler = np.random.permutation(N)
Xtrain = X[shuffler[:Ntrain]]
ytrain = y[shuffler[:Ntrain]]
Xtest = X[shuffler[Ntrain:]]
ytest = y[shuffler[Ntrain:]]


#training a naviebayes classfication
class NevieBayes:
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.numclasses = 3

#calculate the pror probability
    def _get_prior(self,label):
#obtain all counts
        numclasses = self.numclasses
        key = np.unique(ytrain)
        results = {}
        for f in key:
            v = ytrain[ytrain == f].size
            results[f] = v
        prior = np.array([results[i] /(N*0.8) for i in range(0,numclasses)])
        return prior
    # def _get_prior(self):
    #     key = np.unique(ytrain)
    #     cnt = Counter(label)
    #     prior = np.array([cnt[i] / len(label) for i in range(len(cnt))])
    #     return prior

#calculate the means of training data
    #also can use pandas
    def _get_avgs(self,data: Xtrain, label: ytrain)-> ndarray:
        return np.array([data[label == i].mean(axis=0)
                         for i in range(self.numclasses)])
#get the variances of traning data
    def _get_vars(self, data: Xtrain, label: ytrain) -> ndarray:

        return np.array([data[label == i].var(axis=0)
                         for i in range(self.numclasses)])
    #fit function training！！！

    def fit(self, data: Xtrain, label: ytrain)-> ndarray:

        # Calculate prior probability.
        self.prior = self._get_prior(label)
        # Count number of classes.
        self.numclasses = 3
        # Calculate the mean
        self.avgs = self._get_avgs(data, label)
        # Calculate the variance.
        self.vars = self._get_vars(data, label)

        # get the posterior
    def _get_posterior(self, row: Xtrain) -> ndarray:
        return (1 / sqrt(2 * pi * self.vars) * exp(
                -(row - self.avgs) ** 2 / (2 * self.vars))).prod(axis=1)
    def predict_prob(self, data: Xtest) -> ndarray:
#Get the probability of label.

        # Caculate the joint probabilities of each feature and each class.
        likelihood = np.apply_along_axis(self._get_posterior, axis=1, arr=data)
        probs = self.prior * likelihood
        # Scale the probabilities
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    def predict(self, data: Xtest) -> ndarray:
        #Get the prediction of label.

        # Choose the class which has the maximum probability
        return self.predict_prob(data).argmax(axis=1)
if __name__ == "__main__":
        nbc=NevieBayes()
        nbc.fit(Xtrain, ytrain)
        ytrain_hat=nbc.predict(data=Xtrain)
        yhat = nbc.predict(data=Xtest)
        testaccuracy = np.mean(yhat == ytest)
        print('the actual results:',ytest)
        print('the predict results:',yhat)
        print('the accuracy of training:',accuracy_score(ytrain_hat,ytrain))
        print("the accuracy is testing:",accuracy_score(yhat, ytest))




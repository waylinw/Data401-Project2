from sklearn.externals import joblib 
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LogisticRegressionCV          
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
import operator
from io import StringIO
import csv

class Classifier:
    def __init__(self):
        self.model=joblib.load('tfidftop50K.pkl')
        self.Vect=joblib.load('tfidfmixbigram.pkl')
        self.Selector=joblib.load('tfidfselector.pkl')
        self.betas=self.model.coef_

    def update(self, data):
        reader = csv.reader(StringIO(data))
        correct = 0
        total = 0
        while True:
            for row in reader:
                total += 1
                    # define response variable (controversiality)
                y = int(row[20])

                    # gets the body
                body = row[17]
                
                #transform the body into features with tfidf vectorizer, then the 
                # feature_id list for updating coefs
                x_tfidf = self.Vect.transform(body)
                x_train = self.Selector.transform(x_tfidf)
                
                # here will give you a list of indexes to update in the coef_
                feature_index = x_train.getrow(0).nonzero()[1]
                feature_values = x_train.getrow(0).nonzero()[0]
                
                predict=self.model.intercept_
                for ids in range(len(feature_index)):
                    beta = self.betas[feature_index[ids]]
                    predict += beta * feature_values[ids]
                
                resid = y - (1 / (1 + np.exp(-predict)))
                for ids in feature_index:
                    self.beta[feature_index] += (1 / total) * resid
                
                correct += 1 * (abs(resid) < .5)
            print(correct)
                    # determine whether prediction was correct
                    #correct += 1 * (abs(resid) < .5)
                    # print out the accuracy rate every 1000 iterations
                    #if total % 1000 == 0: print(correct / total)
                    #need a list of 0s and 1s of whether or not unigram/bigram is in model
                    #j=self.Vect.transform(body)
                    #predict=self.model.intercept_

                    # get residual of current prediction
                    # Dictionary has key of feature id, value of lookup
                    # needs to be altered to reflect beta equations
                    #for ids, value in j:
                    #    if ids in dictionary:
                    #        beta = betas[dictionary[ids]]
                    #        predict+=beta*value
                    #resid = y - (1 / (1 + np.exp(-predict)))
                    #for i in j:
                    #    beta[j] += (1 / total) * resid

                    # determine whether prediction was correct
                    #correct += 1 * (abs(resid) < .5)
                    # print out the accuracy rate every 1000 iterations
                    #if total % 1000 == 0: print(correct / total)

    def predict(self, data):
         with open(data, newline="") as f:
            reader = csv.reader(f)
            next(reader) # skip header row
            while True:
                total += 1
                try:
                    row = next(reader)
                except StopIteration:
                    break

                # define response variable (controversiality)
                y = int(row[20])

                # gets the body
                body = row[17]
                
                x_tfidf = self.Vect.transform(body)
                x_train = self.Selector.transform(x_tfidf)
                
                # here will give you a list of indexes to update in the coef_
                feature_index = x_train.getrow(0).nonzero()[1]
                feature_values = x_train.getrow(0).nonzero()[0]
                
                
                ## TO DO: Figure out how to calculate the next value. i think we have to calc the resid like above
                ## and it's controversial when the (abs(resid) < .5)

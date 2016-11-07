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
        self.betas=self.model.coef_[0]

    def update(self, data):
        reader = csv.reader(StringIO(data))
        correct = 0
        total = 0
        for row in reader:
            total += 1
            # define response variable (controversiality)
            y = int(row[20])

            # gets the body
            body = row[17]

            #transform the body into features with tfidf vectorizer, then the 
            # feature_id list for updating coefs
            x_tfidf = self.Vect.transform([body])
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
                self.betas[feature_index] += (1 / total) * resid
                
            correct += 1 * (abs(resid) < .5)

    def predict(self, data):
        reader = csv.reader(StringIO(data))
        predict_result=[]
        for row in reader:
            # gets the body
            body = row[17]

            x_tfidf = self.Vect.transform([body])
            x_train = self.Selector.transform(x_tfidf)

            # here will give you a list of indexes to update in the coef_
            feature_index = x_train.getrow(0).nonzero()[1]
            feature_values = x_train.getrow(0).nonzero()[0]

            predict=self.model.intercept_
            for ids in range(len(feature_index)):
                beta = self.betas[feature_index[ids]]
                predict += beta * feature_values[ids]

            predict_val = (1 / (1 + np.exp(-predict)))
            if (abs(predict_val) < .5):
                predict_result.append(0)
            else:
                predict_result.append(1)
                print('controversial')
        return predict_result
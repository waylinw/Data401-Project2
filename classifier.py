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
        self.Mapping = joblib.load('tfidfselectormapping.pkl')
        #self.Selector=joblib.load('tfidfselector.pkl')
        self.betas=self.model.coef_[0]
        self.intercept = self.model.intercept_
        self.total = 200000

    def update(self, data):
        reader = csv.reader(StringIO(data))
        for row in reader:
            self.total += 1
            # define response variable (controversiality)
            y = int(row[20])

            # gets the body
            body = row[17]

            #transform the body into features with tfidf vectorizer, then the 
            # feature_id list for updating coefs
            learn = self.Vect.transform([body])
            pred = 0
            pred += self.intercept
            for val, idx in tuple(zip(learn.todense()[0, learn.nonzero()[1]].tolist()[0], learn.nonzero()[1])):
                real_idx = self.Mapping.get(idx)
                if real_idx != None:
                    pred += self.betas[real_idx] * val
            
            resid = y - (1 / (1 + np.exp(-pred)))
            
            for val, idx in tuple(zip(learn.todense()[0, learn.nonzero()[1]].tolist()[0], learn.nonzero()[1])):
                real_idx = self.Mapping.get(idx)
                if real_idx != None:
                    self.betas[real_idx] += (1 / self.total) * resid

    def predict(self, data):
        reader = csv.reader(StringIO(data))
        predict_result=[]
        for row in reader:
            # gets the body
            body = row[17]
            
            test = self.Vect.transform([body])
            pred = 0
            pred += self.intercept
            for val, idx in tuple(zip(test.todense()[0, test.nonzero()[1]].tolist()[0], test.nonzero()[1])):
                real_idx = self.Mapping.get(idx)
                if real_idx != None:
                    pred += self.betas[real_idx] * val

            predict_val = (1 / (1 + np.exp(-pred)))
            
            if (abs(predict_val) < .5):
                predict_result.append(0)
            else:
                predict_result.append(1)
                
        return predict_result
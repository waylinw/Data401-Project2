from sklearn.externals import joblib 
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV

from sklearn.cross_validation import cross_val_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import LogisticRegressionCV          
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
import operator
from io import StringIO
import csv
class Classifier:
    def __init__(self):
        self.model=joblib.load('tfidftop50K.pkl')
        self.Vect=joblib.load('tfidfmixbigram.pkl')
        self.Selector=joblib.load('tfidfselector.pkl')

    def update(self, data):
        with open(data, newline="") as f:
            reader = csv.reader(StringIO(data))
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
                betas=self.model.coef_
                #need a list of 0s and 1s of whether or not unigram/bigram is in model
                j=self.Vect.transform(body)
                predict=self.model.intercept_
                # get residual of current prediction
                # Dictionary has key of feature id, value of lookup
                # needs to be altered to reflect beta equations
                for ids, value in j:
                    if ids in dictionary:
                        beta = betas[dictionary[ids]]
                        predict+=beta*value
                resid = y - (1 / (1 + np.exp(-predict)))
                for i in j:
                    beta[j] += (1 / total) * resid

                # determine whether prediction was correct
                correct += 1 * (abs(resid) < .5)
                # print out the accuracy rate every 1000 iterations
                if total % 1000 == 0: print(correct / total)

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
                betas=self.model.coef_
                #need a list of 0s and 1s of whether or not unigram/bigram is in                 
                j=Vect.transform(body)
                predict=self.model.intercept_
                # get residual of current prediction
                # Dictionary has key of feature id, value of lookup
                # needs to be altered to reflect beta equations
                for ids, value in j:
                    if ids in dictionary:
                        beta = betas[dictionary[ids]]
                        predict+=beta*value
                resid = y - (1 / (1 + np.exp(-predict)))
                for i in j:
                    beta[j] += (1 / total) * resid

                # determine whether prediction was correct
                correct += 1 * (abs(resid) < .5)
                # print out the accuracy rate every 1000 iterations
                if total % 1000 == 0: print(correct / total)

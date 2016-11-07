from sklearn.externals import joblib 
from io import StringIO
import csv
,class Classifier:

    def __init__(self):
        self.model=joblib.dump(model,'tfidftop50K.pkl',compress=9)
        self.Vect=joblib.dump(vect,'tfidfmxbigram.pkl',compress=9)

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
                betas=model.coef_
                #need a list of 0s and 1s of whether or not unigram/bigram is in model
                j=Vect.transform(body)
                predict=model.intercept_
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

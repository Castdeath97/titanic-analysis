# Imports
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LogisticRegression
import numpy as np

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as met

# Data Loading

# load using pandas to get pandas dfs
trainDf = pd.read_csv('train.csv')
testDf = pd.read_csv('test.csv')

# Data Cleaning Functions

# Drops unrequired fields (id, ticket, cabin and name)
def dropFields(df):
    df = df.drop('PassengerId', axis=1)
    df = df.drop('Ticket', axis=1)
    df = df.drop('Cabin', axis=1)
    df = df.drop('Name', axis=1)
    return(df)

# replaces a given NA ages with averages
def fixNaAges(df, sex):
    # find Df for chosen Sex and record average age
    sexDf = df[df.Sex == sex] 
    avgSexAge = int(round(sexDf.Age.mean()))
    
    # Update Age NAs with the calculated average 
    sexDf.fillna({'Age':avgSexAge}, inplace = True)
    
    # copy updated Df for chosen sex to df and return
    df[df.Sex == sex] = sexDf
    return(df)

# prints counts for the two dfs
def printCounts(trainDf, testDf, caption):
    print("Train Counts: " + caption)
    print(trainDf.count())
    print("Test Counts: " + caption)
    print(testDf.count())
    
# Data cleaning

# Print inital counts 
printCounts(trainDf, testDf, "Initial")

# Drop Fields
cTrainDf = dropFields(trainDf)
cTestDf = dropFields(testDf)
printCounts(cTrainDf, cTestDf, "After dropping fields")

# Fix age NAs with sex average ages
cTrainDf = fixNaAges(cTrainDf, "male")
cTrainDf = fixNaAges(cTrainDf, "female")
cTestDf = fixNaAges(cTestDf, "male")
cTestDf = fixNaAges(cTestDf, "female")
printCounts(cTrainDf, cTestDf, "After fixing Age NAs")

# Drop any left NAs (a few from embarkment and fare)
cTestDf = cTestDf.dropna(axis = 0)
cTrainDf = cTrainDf.dropna(axis = 0)
printCounts(cTrainDf, cTestDf, "After dropping rows with NAs")


# Change categorical fields with strings to ints 
# for fitting

cTestDf['Sex'] = cTestDf['Sex'].replace(['female'],0)
cTestDf['Sex'] = cTestDf['Sex'].replace(['male'],1)

cTrainDf['Sex'] = cTrainDf['Sex'].replace(['female'],0)
cTrainDf['Sex'] = cTrainDf['Sex'].replace(['male'],1)

# Add new binary variables for 2 value embarked 
cTestDf['EmbarkedS'] = cTestDf['Embarked']
cTestDf['EmbarkedS'] = cTestDf['EmbarkedS'].replace(['S'],1)
cTestDf['EmbarkedS'] = cTestDf['EmbarkedS'].replace(['Q'],0)
cTestDf['EmbarkedS'] = cTestDf['EmbarkedS'].replace(['C'],0)

cTestDf['EmbarkedQ'] = cTestDf['Embarked']
cTestDf['EmbarkedQ'] = cTestDf['EmbarkedQ'].replace(['S'],0)
cTestDf['EmbarkedQ'] = cTestDf['EmbarkedQ'].replace(['Q'],1)
cTestDf['EmbarkedQ'] = cTestDf['EmbarkedQ'].replace(['C'],0)

#cTestDf['EmbarkedC'] = cTestDf['Embarked']
#cTestDf['EmbarkedC'] = cTestDf['EmbarkedC'].replace(['S'],0)
#cTestDf['EmbarkedC'] = cTestDf['EmbarkedC'].replace(['Q'],0)
#cTestDf['EmbarkedC'] = cTestDf['EmbarkedC'].replace(['C'],1)


cTrainDf['EmbarkedS'] = cTrainDf['Embarked']
cTrainDf['EmbarkedS'] = cTrainDf['EmbarkedS'].replace(['S'],1)
cTrainDf['EmbarkedS'] = cTrainDf['EmbarkedS'].replace(['Q'],0)
cTrainDf['EmbarkedS'] = cTrainDf['EmbarkedS'].replace(['C'],0)

cTrainDf['EmbarkedQ'] = cTrainDf['Embarked']
cTrainDf['EmbarkedQ'] = cTrainDf['EmbarkedQ'].replace(['S'],0)
cTrainDf['EmbarkedQ'] = cTrainDf['EmbarkedQ'].replace(['Q'],1)
cTrainDf['EmbarkedQ'] = cTrainDf['EmbarkedQ'].replace(['C'],0)

#cTrainDf['EmbarkedC'] = cTrainDf['Embarked']
#cTrainDf['EmbarkedC'] = cTrainDf['EmbarkedC'].replace(['S'],0)
#cTrainDf['EmbarkedC'] = cTrainDf['EmbarkedC'].replace(['Q'],0)
#cTrainDf['EmbarkedC'] = cTrainDf['EmbarkedC'].replace(['C'],1)

cTrainDf = cTrainDf.drop(columns=['Embarked']) # drop original, no longer needed
cTestDf = cTestDf.drop(columns=['Embarked']) # drop original, no longer needed

# separate predictors from response and scale predictors

testPred = cTestDf.loc[:, cTestDf.columns != 'Survived']
testRes = cTestDf['Survived']
trainPred = cTrainDf.loc[:, cTrainDf.columns != 'Survived']
trainRes = cTrainDf['Survived']

mapper = DataFrameMapper([(trainPred.columns, pre.StandardScaler())])
scaledFeatures = mapper.fit_transform(trainPred.copy(), len(trainPred.columns))
sTrainPred = pd.DataFrame(scaledFeatures, index=trainPred.index, columns=trainPred.columns)

mapper = DataFrameMapper([(testPred.columns, pre.StandardScaler())])
scaledFeatures = mapper.fit_transform(testPred.copy(), len(testPred.columns))
sTestPred = pd.DataFrame(scaledFeatures, index=testPred.index, columns=testPred.columns)

## print metrics (Accuracy, confusion, f1)
def printMetrics(prediction):
    print('accuracy', met.accuracy_score(testRes, prediction))
    print()
    print(met.confusion_matrix(testRes, prediction))
    print()
    print('f1', met.f1_score(testRes, prediction))
    
# Fit and prediction
clf = LogisticRegression().fit(sTrainPred, trainRes)
prediction = clf.predict(sTestPred)

# create table for coefs
coefs = pd.concat([pd.DataFrame(trainPred.columns),
                 pd.DataFrame(np.transpose(clf.coef_))],
                axis = 1)
coefs.loc[len(coefs)] = ['intercept', clf.intercept_[0]]
print(coefs) # print table

# Metrics (predtion vs Test response)
printMetrics(prediction)

# Code adapted from https://github.com/yuguan1/example-ML-code

# Set the parameters by cross-validation  ['linear', 'rbf', 'sigmoid','poly'],

parametersRbf = [{'kernel': ['rbf'],
                'gamma': [1e-4, 0.01, 0.1, 0.5],
                'C': [1, 10, 100, 1000]}]

parametersLin = [{'kernel': ['linear'],
                'C': [1, 10, 100, 1000]}]

parametersSig = [{'kernel': ['sigmoid'],
                'gamma': [1e-4, 0.01, 0.1, 0.5],
                'C': [1, 10, 100, 1000]}]

parametersPoly = [{'kernel': ['poly'],
                'C': [1, 10, 100, 1000]}]

# Tune parameters using 5 fold cross validation 

def tuneParams(parameters):
    print("# Tuning hyper-parameters")
    clf = GridSearchCV(SVC(), parameters, cv=5)
    print("# HERE 1")
    clf.fit(sTrainPred, trainRes)
    print("# HERE 2")

    print('best parameters:')
    print(clf.best_params_)
    print('-------------------------------------')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        
tuneParams(parametersSig)
tuneParams(parametersLin)
tuneParams(parametersRbf)
tuneParams(parametersPoly)
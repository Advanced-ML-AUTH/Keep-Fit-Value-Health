# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 14:31:01 2023

@author: Fik
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:42:50 2023

@author: Fik
"""

'''
    Dataset's links:
        
        https://www.kaggle.com/code/mpwolke/obesity-levels-life-style
        https://www.kaggle.com/code/mpwolke/obesity-levels-life-style/input
    
    Links used:
        
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        
        https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
        https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/
        https://medium.com/@rithpansanga/logistic-regression-and-regularization-avoiding-overfitting-and-improving-generalization-e9afdcddd09d
        
       
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt


def exploreData(df):
    
    print(df.head(), '\n')
    print(df.info(), '\n')
    print(df.describe(include = 'all').transpose(), '\n')
    print(df.isnull().sum(), '\n')
    
    for col in df.columns:
        print(col,'\n', df[col].dtype,'\n', df[col].unique())
    print('\n')


def dropAnyUnnamedCols(df):
    
    for col in df.columns:
        if 'Unnamed' in col:
            df.drop([col], axis=1, inplace=True)
            

def separate_xValues_yValues(df_xValues, y_name, flag_to_numpy=False, flag_xValNames=False):

    df_yValue = df_xValues[y_name]
    df_xValues.drop(labels=y_name, axis=1, inplace=True)
    #print(df_yValue.name)
    #print(df_xValues.columns)
    
    if flag_to_numpy:#flag 1 or 0
        if flag_xValNames:
            return df_xValues.to_numpy(), df_yValue.to_numpy(), df_xValues.columns
        return df_xValues.to_numpy(), df_yValue.to_numpy()
    else:
        #print(flag_to_numpy)
        return df_xValues, df_yValue    


def extract_for_test(X_train, X_test, y_train, y_test, n, l):
 
    return X_train.head(n), X_test.head(l), y_train.head(n), y_test.head(l) 


def scaleCols(df, std_colsToScale_list, minMax_colsToScale_list):
    #https://www.geeksforgeeks.org/python-scaling-numbers-column-by-column-with-pandas/
    std_scaler = StandardScaler()
    df[std_colsToScale_list] = std_scaler.fit_transform(df[std_colsToScale_list])
    
    minMax_scaler = MinMaxScaler()
    df[minMax_colsToScale_list] = minMax_scaler.fit_transform(df[minMax_colsToScale_list])

    return df


def createParamGrid(numEstim_list, maxDepth_list, start_stop_flag, min_samples_split_list = [], min_samples_leaf_list = []):
    
    if start_stop_flag:
        # Number of trees in random forest
        n_estimators_list = [int(x) for x in np.linspace(start = numEstim_list[0], stop = numEstim_list[1], num = numEstim_list[2])]
        #print(type(n_estimators_list ))
        
        # Maximum number of levels in tree
        maxDepth_list = [int(x) for x in np.linspace(maxDepth_list[0], maxDepth_list[1], num = maxDepth_list[2])]
        #print(type(maxDepth_list))
    else:
        n_estimators_list = numEstim_list
        maxDepth_list = maxDepth_list
    
   
    if not min_samples_split_list:
        # Minimum number of samples required to split a node
        min_samples_split_list = [2, 5, 10]
    
    if not min_samples_leaf_list:
        # Minimum number of samples required at each leaf node
        min_samples_leaf_list = [1, 2, 4]
    
    # Number of features to consider at every split
    max_features_list = ['log2', 'sqrt']
    
    # Method of selecting samples for training each tree
    bootstrap_list = [True, False]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators_list, 'max_features': max_features_list,
                   'max_depth': maxDepth_list, 'min_samples_split': min_samples_split_list,
                   'min_samples_leaf': min_samples_leaf_list, 'bootstrap': bootstrap_list}

    #print(param_grid , type(param_grid))
    print(param_grid)
    
    return param_grid


def gridSearchRandForest_multiClass(X_train, y_train, param_grid,  cv_num,  avg_str,  iter_num=0, verb = 2):
    '''
    verbose: int
        >1 : the computation time for each fold and parameter candidate is displayed;
        >2 : the score is also displayed;
        >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
    
    n_jobsint, default=None
        Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
    '''
    random_state = 42
    scorer = metrics.make_scorer(metrics.f1_score, average = avg_str) #https://stats.stackexchange.com/questions/431022/error-while-performing-multiclass-classification-using-gridsearch-cv
   
    clf = RandomForestClassifier(random_state = random_state)
    clf.fit(X_train, y_train)
    
    if iter_num:
        clf_custom = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_iter = iter_num, scoring = scorer,  cv = cv_num, verbose = verb, n_jobs=-1, return_train_score=True)
    else:
        clf_custom = GridSearchCV(estimator = clf, param_grid = param_grid, scoring = scorer,  cv = cv_num, verbose = verb, n_jobs=-1,  return_train_score=True)
        
    
    clf_custom.fit(X_train, y_train)
    clf_custom_best_params = clf_custom.best_params_
    print("Best: %f using %s" % ( clf_custom.best_score_,  clf_custom.best_params_))
    
    return clf, clf_custom.best_estimator_, clf_custom_best_params 


def val_model(X_test, y_test, clf, avg_str):
    
    y_pred = clf.predict(X_test)
    
    # Model Accuracy, how often is the classifier correct?
    acc = metrics.accuracy_score(y_test, y_pred)

    prec = metrics.precision_score(y_test, y_pred, average=avg_str)
    recall = metrics.recall_score(y_test, y_pred, average=avg_str)
    f1 = metrics.f1_score(y_test, y_pred, average=avg_str)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    
    print(" Accuracy val: ", acc, " Prec val: ", prec, " Recall val: ", recall, " F1: ", f1, '\n')
    return acc, prec, recall, f1

"""
Load the dataframe and perform necessary preprocessing steps.
Generate a plot that illustrates the significance of each column.
Re-train the model using the modified dataset.
"""
def findColumnSignificance(best_model, X):
    importances = best_model.feature_importances_
    feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title("Feature Importances")
    plt.xlabel("Columns")
    plt.ylabel("Importance")
    plt.show()

    newDf = pd.read_csv('data/obesity.csv', sep=';')
    dropAnyUnnamedCols(newDf)
    newDf= scaleCols(newDf, std_colsToScale_list=['Height','Weight'], minMax_colsToScale_list= ['Age'])
    newDf = dropInsignificanceCols(newDf)
    best_model, X = trainModel(newDf)
    print(newDf)

    return best_model, X


def trainModel(df):
    targVarName='NObeyesdad'

    # Split independent varables from target/dependent variable
    X,y = separate_xValues_yValues(df, targVarName)
    print(X)
    print(y)
    # Split to training and testing sets
    testSize=0.2; randState=0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randState)
    
    #'''
    X_train, X_test, y_train, y_test  = extract_for_test( X_train, X_test, y_train, y_test, n = 1000, l = 200)
    #'''

    avg_str ='micro' # Set number of cross validation folds 
    
    best_params = {
    'n_estimators': 1800,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'log2',
    'max_depth': 40,
    'bootstrap': False,
    'random_state': 42
    }

    #train model using the best params
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)


    acc_randSearch, prec_randSearch, recall_randSearch, f1_randSearch = val_model(X_test, y_test, best_model, avg_str)

    return best_model, X

def dropInsignificanceCols(df):
    columns_to_remove = ['SMOKE', 'SCC', 'Automobile', 'Bike', 'Motorbike', 'Transportation', 'Walking', 'FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF',  'CALC', 'family_history', 'FAVC']
    df = df.drop(columns_to_remove, axis=1)
    return df

def main():
    
    # Configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
    
    df = pd.read_csv('C:/Users/kyriaki.potamopoulou/Documents/DWS/ML/data/obesity.csv', sep=';')
    print(df)
    dropAnyUnnamedCols(df)

    df= scaleCols(df, std_colsToScale_list=['Height','Weight'], minMax_colsToScale_list= ['Age'])
    print(df.head(2))
    
    exploreData(df)

    best_model, X = trainModel(df)

    best_model, X = findColumnSignificance(best_model, X)

    #save model
    joblib.dump(best_model, 'C:/Users/kyriaki.potamopoulou/Documents/DWS/ML/obesity.joblib')

            
    
if __name__ == "__main__":
#   https://www.quora.com/When-I-import-my-module-in-python-it-automatically-runs-all-of-the-defined-functions-inside-of-it-How-do-I-prevent-it-from-auto-executing-my-functions-but-still-allow-me-to-call-them-in-my-main-script

    '''
    try:
        main()
    except:
        pass
    '''
    main()
    #'''

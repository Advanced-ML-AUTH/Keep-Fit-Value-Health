# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 08:42:50 2023

@author: Fik
"""

'''
    Dataset's links:
        
        https://www.kaggle.com/code/mpwolke/obesity-levels-life-style
        https://www.kaggle.com/code/mpwolke/obesity-levels-life-style/input
    
    Links consulted:
        
        https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/
    
    Links used:
        
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        
        https://www.geeksforgeeks.org/python-scaling-numbers-column-by-column-with-pandas/
        https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8      
        https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
        https://www.projectpro.io/recipes/optimize-hyper-parameters-of-logistic-regression-model-using-grid-search-in-python
        
        https://stats.stackexchange.com/questions/431022/error-while-performing-multiclass-classification-using-gridsearch-cv     
       
'''
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn import metrics
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

global_file = None


def open_file(filename):
    global global_file
    global_file = open(filename, "w+")

    
def write_to_file(data):
    global global_file
    if global_file:
        global_file.write(data + "\n")


def close_file():
    global global_file
    if global_file:
        global_file.close()
        global_file = None


def drop_annyUnnamed_cols(df):
    
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


def scale_cols(df, std_colsToScale_list, minMax_colsToScale_list):
    #https://www.geeksforgeeks.org/python-scaling-numbers-column-by-column-with-pandas/
    std_scaler = StandardScaler()
    df[std_colsToScale_list] = std_scaler.fit_transform(df[std_colsToScale_list])
    
    minMax_scaler = MinMaxScaler()
    df[minMax_colsToScale_list] = minMax_scaler.fit_transform(df[minMax_colsToScale_list])

    return df
 

def grid_search_logReg_multiClass(X_train, y_train, param_dict, cv, avg_str, iter_num=0, maxIter=1000, verb=2, randState=0):
    '''
    verbose: int
        >1 : the computation time for each fold and parameter candidate is displayed;
        >2 : the score is also displayed;
        >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
    
    n_jobsint, default=None
        Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
    '''
    if randState:
        clf = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=maxIter, random_state=randState)
    else:
        clf = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=maxIter)        
    # define the model evaluation procedure
    
    scorer = metrics.make_scorer(metrics.f1_score, average = avg_str) #https://stats.stackexchange.com/questions/431022/error-while-performing-multiclass-classification-using-gridsearch-cv
   
    if iter_num:
        clf_custom = RandomizedSearchCV(clf, param_dict, n_iter = iter_num, cv=cv, scoring=scorer, verbose=verb, n_jobs=-1, return_train_score=True)
    else:

        clf_custom = GridSearchCV(clf, param_dict, cv=cv, scoring=scorer, verbose=0, n_jobs=-1, return_train_score=True)
    
    clf_custom.fit(X_train, y_train)
    clf_custom_best_params = clf_custom.best_params_
    print("Best: %f using %s" % ( clf_custom.best_score_,  clf_custom.best_params_))
    write_to_file( "Best: %f using %s" % ( clf_custom.best_score_,  clf_custom.best_params_) )

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
    write_to_file("Accuracy val: "+ str(acc)+ " Prec val: "+ str(prec)+ " Recall val: "+ str(recall)+ " F1: "+ str(f1)+ '\n')
    return acc, prec, recall, f1


def plot_learning_curve(estimator, X, y, cv, scoring):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', color='b')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='b')
    plt.plot(train_sizes, val_mean, label='Validation Score', color='r')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='r')

    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()

    
def main():
    # Configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
    
    df = pd.read_csv('obesity.csv')
    
    drop_annyUnnamed_cols(df)
    #exploreData(df)
    print(df.head(2))
    df= scale_cols(df, std_colsToScale_list=['Height','Weight'], minMax_colsToScale_list= ['Age'])
    print(df.head(2))
    
    targVarName='NObeyesdad'
    
    # Split independent varables from target/dependent variable
    X,y = separate_xValues_yValues(df, targVarName)
    
    open_file('obesity_log_reg.txt')
    # Split initial dataset to training and testing sets
    testSize=0.25; randState=0 
    write_to_file('testSize = '+str(testSize)+' randState = '+str(randState))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randState)
    
    
    # Grid Searches 

    param_dict = {'C':np.logspace(-4, 4, 50), 'penalty': ['l1', 'l2', 'elasticnet']} # Set dictionary with logistic regression parameters
    cv_num = 10; avg= 'micro' # Set number of cross validation folds and average since our model's metric is F1
    write_to_file('Dictionary with Logistic Regression Parameters: ' + json.dumps({'C': np.logspace(-4, 4, 50).tolist(), 'penalty': ['l1', 'l2', 'elasticnet']}) 
                  + '\nNumber of cross validation folds: ' + str(cv_num) + "\nAverage for our model\'s evaluation metric (F1): "  + avg + '\n')
    
    


    # Exaustive Grid Search
    write_to_file('Exaustive Grid Search: exhaustively searches through a predefined set of hyperparameters\n')
    clf, clf_ex, clf_ex_params = grid_search_logReg_multiClass(X_train, y_train, param_dict, cv_num, avg) # base classifier, best cdlassifier of Exaustive Grid Search, best classifier's parameters 
    acc_exSearch, prec_exSearch, recall_exSearch, f1_exSearch = val_model(X_test, y_test, clf_ex, avg)

    plot_learning_curve(clf_ex, X_train, y_train, cv=cv_num, scoring='f1_micro')

    # Random Grid Search 
    numIter = 25
    write_to_file('Random Grid Search: randomly samples hyperparameter combinations from a predefined search space\nThe '\
                  'fixed number of random hyperparameter combinations it selects to check is:'+str(numIter)+
                  '\nIt evaluates them using cross-validation')    
    clf, clf_rand, clf_rand_params = grid_search_logReg_multiClass(X_train, y_train, param_dict, cv_num, avg, iter_num = numIter) # base classifier, best cdlassifier of Exaustive Grid Search, best classifier's parameters 
    acc_randSearch, prec_randSearch, recall_randSearch, f1_randSearch = val_model(X_test, y_test, clf_ex, avg)            
    
    plot_learning_curve(clf_rand, X_train, y_train, cv=cv_num, scoring='f1_micro')
    
    close_file()
    
    
    
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
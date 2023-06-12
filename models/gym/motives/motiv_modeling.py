'''
    Links used:
        https://www.section.io/engineering-education/multi-label-classification-with-scikit-multilearn/
        https://www.tutorialspoint.com/add-a-key-value-pair-to-dictionary-in-python
        
    Links consulted:
        https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

    Dataset Link:
        https://www.kaggle.com/datasets/nithilaa/fitness-analysis?resource=download

'''

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


from sklearn.metrics import classification_report, zero_one_loss, f1_score


def exploreData(df, colFlag=False):
    
    print(df.head(), '\n')
    print(df.info(), '\n')
    print(df.describe(include = 'all').transpose(), '\n')
    print(df.isnull().sum(), '\n')
    
    if colFlag:
        for col in df.columns:
            print(col,'\n', df[col].dtype,'\n', df[col].unique())
    print('\n')


def dropAnyUnnamedCols(df):
    
    for col in df.columns:
        if 'Unnamed' in col:
                df.drop([col], axis=1, inplace=True)


def separate_X_y_onColName(df, keyWord):
    y_colNames_list = []
    for col in df.columns:
        if keyWord in col:
            y_colNames_list.append(col)
    y = df.loc[:,y_colNames_list]
    X = df.drop(y_colNames_list, axis=1)
    return X,y


def get_models_nn_paramGrids():
    models_dict = {'br_nb_clf' : BinaryRelevance(MultinomialNB()), 
                   'chain_nb_clf': ClassifierChain(MultinomialNB()),
                   'lp_nb_clf': LabelPowerset(MultinomialNB()),
                   'multi_logreg': MultiOutputClassifier(LogisticRegression(max_iter=10000)),
                   'multi_linear_svm':  MultiOutputClassifier(LinearSVC(max_iter=10000)),
                   'multi_svm':  MultiOutputClassifier(SVC()),
                   'rf' : RandomForestClassifier(),
                   }

    mn_nb_pg = {'classifier__alpha': [0.1, 1.0, 5, 10.0],  # Smoothing parameter for MultinomialNB
                'classifier__fit_prior': [True, False]  # Whether to learn class prior probabilities or not
                } # multi-nomial naive bayes parameters grid
    
    mo_lr_pg = {'estimator__C': [0.1, 1.0, 5, 10.0],  # Regularization parameter for LogisticRegression
                'estimator__solver': ['liblinear', 'lbfgs'],  # Solver algorithm for LogisticRegression
                'estimator__penalty': ['l1', 'l2']  # Penalty type for LogisticRegression
                }# multi-output logistiv regression parameters grid
    
    mo_lsvc_pg = { 'estimator__C': [0.1, 1.0, 10.0],  # Regularization parameter
                  'estimator__penalty': ['l2']  # Penalty type
                  } # multi-output linear svm parameters grid
    
    mo_svm_pg = {'estimator__C': [0.1, 1.0, 5, 10.0],  # Regularization parameter for SVC
                 'estimator__kernel': ['linear', 'rbf'],  # Kernel function for SVC
                 'estimator__gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf' kernel
                 }  # multi-output linear svm parameters grid
    
    rf_pg = {'n_estimators': [100, 200, 300],  # Number of trees in the forest
             'max_depth': [None, 5, 10],  # Maximum depth of the tree
             'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
             'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            }
        
    models_nn_param_dod = {'br_nb_clf': mn_nb_pg, 'chain_nb_clf': mn_nb_pg, 'lp_nb_clf': mn_nb_pg,
                           'multi_logreg': mo_lr_pg, 
                           'multi_linear_svm': mo_lsvc_pg,
                           'multi_svm' : mo_svm_pg,
                           'rf': rf_pg,
                          }
    
    
    return models_dict, models_nn_param_dod


def apply_gridSearch(models_dict, models_nn_param_dod, X_train, y_train, X_test, y_test):

    s=''
    for model_name, model in models_dict.items():
        grid_search = GridSearchCV(model, models_nn_param_dod[model_name], cv=10)
        grid_search.fit(X_train, y_train)

        # Predict on the test data
        pred_labels = grid_search.best_estimator_.predict(X_test)
        best_param_string = '{' + ', '.join(f'{key}: {value}' for key, value in grid_search.best_params_.items()) + '}'

        s+= '\nEvaluating {}'.format(model_name)+'\n'+'Best Parameters: '+best_param_string +'\n'+classification_report(y_test, pred_labels)+'\nTest accuracy = {}'.format((1 - zero_one_loss(y_test, pred_labels))) +'\n\n'

    print(s)


def apply_pca(X, n_comp):
    pca = PCA(n_components=n_comp)
    principalComponents = pca.fit_transform(X)
    pcX = pd.DataFrame(data = principalComponents , columns = ['pc'+str(i+1) for i in range(n_comp)])
    return pcX


def scale_X_train_X_test(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def main():
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
 
    csvName = 'fitness.csv'
    df = pd.read_csv(csvName)
    dropAnyUnnamedCols(df)

    exploreData(df)

    X,y = separate_X_y_onColName(df, 'motiv_')
    #print(X.head()); print(y.head())

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    models_dict, models_nn_param_dod = get_models_nn_paramGrids()
    
    #apply_gridSearch(models_dict, models_nn_param_dod, X_train, y_train, X_test, y_test)

    n=10
    pcX = apply_pca(X, n)
    
    X_train,X_test,y_train,y_test = train_test_split(pcX,y,test_size=0.2,random_state=42)
    X_train, X_test = scale_X_train_X_test(X_train, X_test)

    apply_gridSearch(models_dict, models_nn_param_dod, X_train, y_train, X_test, y_test)





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
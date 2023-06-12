# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:18:44 2023

@author: Fik
"""

'''
    Dataset's links:
        https://www.kaggle.com/code/shailesh2692/body-fat-prediction-with-rmse-4-186/input
        https://www.kaggle.com/code/shailesh2692/body-fat-prediction-with-rmse-4-186/notebook
    
    Links used:
        https://www.geeksforgeeks.org/how-to-rename-columns-in-pandas-dataframe/
        https://datatofish.com/round-values-pandas-dataframe/
        https://sparkbyexamples.com/pandas/pandas-convert-column-to-int/
       
'''

import pandas as pd

def exploreData(df):
    
    print(df.head(), '\n')
    print(df.info(), '\n')
    print(df.describe(include = 'all').transpose(), '\n')
    print(df.isnull().sum(), '\n')
    
    for col in df.columns:
        print(col,'\n', df[col].dtype,'\n', df[col].unique())
    print('\n')

        
def main():
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
    
    '''    
    csvName_list = ['bodyfat.csv', 'ObesityDataSet_raw_and_data_sinthetic.csv']
    
    for csvName in csvName_list:
        df = pd.read_csv(csvName)
        #print(df.columns)
        exploreData(df)
    '''
    
    csvName = 'ObesityDataSet_raw_and_data_sinthetic.csv'
    df = pd.read_csv(csvName)
    exploreData(df)
    
    print(df.columns)
    
    '''
    Attributes related with eating habits are:
        SEX : 0 - Female, 1 Male
        Frequent consumption of high caloric food (FAVC), --> 0/1 yes
        Frequency of consumption of vegetables (FCVC), --> int
        Number of main meals (NCP), --> int
        Consumption of food between meals (CAEC), --> int 
        Consumption of water daily (CH20), --> int
        Consumption of alcohol (CALC). --> maping

    Attributes related with the physical condition are:
        Calories consumption monitoring (SCC), --> 0/1
        Physical activity frequency (FAF), --> int
        Time using technology devices (TUE), --> int or double (hours)
        Transportation used (MTRANS), --> 0/1
    '''
    
    catVar_list = ['Gender', 'family_history_with_overweight', 'FAVC',  'SMOKE',  'SCC', 'MTRANS'] 
    df = pd.get_dummies(df, columns = catVar_list ) # https://stackoverflow.com/questions/62017554/pd-get-dummies-only-keep-dummy-value-name-as-dummy-column-name
    #print(df.columns)
    df.drop(['Gender_Female', 'family_history_with_overweight_no', 'FAVC_no', 'SMOKE_no', 'SCC_no'], axis = 1, inplace =True)
    #print(df.columns)

    rename_dict = {'Gender_Male':'Gender', 'family_history_with_overweight_yes':'family_history', 'FAVC_yes':'FAVC', 'SMOKE_yes':'SMOKE', 'SCC_yes':'SCC', 'MTRANS_Automobile': 'Automobile','MTRANS_Bike':'Bike', 'MTRANS_Motorbike':'Motorbike', 'MTRANS_Public_Transportation':'Transportation', 'MTRANS_Walking': 'Walking'}
    df.rename(columns = rename_dict, inplace = True)
    # https://www.geeksforgeeks.org/how-to-rename-columns-in-pandas-dataframe/ 
    
    ordVar_dod = {'CAEC':{'Sometimes':1, 'Frequently':2, 'Always':3, 'no':0}, 
                  'CALC':{'no':0, 'Sometimes':1, 'Frequently':2, 'Always':3},
                  'NObeyesdad':{'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III': 6}
                 }
    for col in ordVar_dod:
        df[col] = df[col].map(ordVar_dod[col]) 

        

    
    intVar_list =['Age', 'FCVC', 'NCP', 'CH2O', 'FAF']
    for col in intVar_list:
        df[col] = df[col].round().astype('int') 
        # https://www.statology.org/pandas-round-column/
        # https://sparkbyexamples.com/pandas/pandas-convert-column-to-int/
    
    floatVar_list = ['Height', 'Weight', 'TUE']
    for col in floatVar_list:
        df[col]=df[col].round(2)
    
    exploreData(df)
    
    df.to_csv('obesity.csv')

    
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


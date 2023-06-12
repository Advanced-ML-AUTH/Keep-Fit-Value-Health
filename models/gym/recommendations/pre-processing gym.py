# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:08:57 2023

@author: Fik
"""

'''
    Dataset's links:
        https://www.kaggle.com/datasets/niharika41298/gym-exercise-data
    
    Links used:
       
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
 
    csvName = 'megaGymDataset.csv'
    df = pd.read_csv(csvName)
    exploreData(df)#; print(df.columns)
    
    # Drop 'Unnamed: 0', which is an id column and 'RatingDesc' column, which has only two values: 'Average' and NA 
    df.drop(['Unnamed: 0', 'RatingDesc'], axis=1, inplace=True)
    
    exploreData(df)
    
    # Drop rows with NA's in 'Desc' column
    df = df.dropna(subset=['Desc'], axis=0) # https://www.digitalocean.com/community/tutorials/pandas-dropna-drop-null-na-values-from-dataframe
    # Replace the NA values, in  'Rating' column, with 0
    df['Rating'].fillna(0, inplace=True)
    df['Level'] = df['Level'].map({'Intermediate':1, 'Beginner':0,'Expert':2}) 
    
    df.to_csv('gym_no_dummies.csv')

    ## Remove rows which have 'Powerlifting', 'Strongman', 'Olympic Weightlifting' values in Type column
    # Specify the specific values you want to remove
    special_values_colType_list = ['Powerlifting', 'Strongman', 'Olympic Weightlifting']
    # Create a boolean mask based on the condition
    df_filtered = df[~df['Type'].isin(special_values_colType_list)]
    # Use the mask to filter out the rows with the specific values
    df_filtered.to_csv('gym_no_dummies_noPowerStrongmanOW.csv')
    
    
    exploreData(df)
    catVar_list = ['Type', 'BodyPart', 'Equipment']
    rename_list = []
    for col in catVar_list:
        rename_list+=list(df[col].unique())

    df = pd.get_dummies(df, columns = catVar_list)#, drop_first=True ) # https://stackoverflow.com/questions/62017554/pd-get-dummies-only-keep-dummy-value-name-as-dummy-column-name
    # https://dataindependent.com/pandas/pandas-get-dummies-pd-get_dummies/ 
        
    # remove vars: 'Title', 'Desc', 'Level', 'Rating'
    dummies_list = list(df.columns)[4:]#; print(dummies_list); print(len(dummies_list),len(rename_list))
    rename_dict = dict( zip(dummies_list, rename_list) )
    # https://careerkarma.com/blog/python-convert-list-to-dictionary/
    df.rename(columns = rename_dict, inplace = True)
    

    
    exploreData(df)
    
    df.to_csv('gym.csv')    

    
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


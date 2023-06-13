import pandas as pd
import numpy

def removeZero(df): 
    print(df.head)

    #calculate the percentage of values equal to 0 in each column of the DataFrame. 
    zeroPercentage = df.apply(lambda x: ((x.astype(str).eq('0') | x.astype(str).eq('0.0 g') | x.astype(str).eq('0.000 g') | x.astype(str).eq('0.00 mg') |  x.astype(str).eq('NaN')).sum() / len(x)) * 100)

    #kee[] only the columns that have a percentage of zero values equal to or lower than 95%
    columnToKeep = zeroPercentage[zeroPercentage <= 95].index
    dfWithoutZero = df[columnToKeep]

    print(dfWithoutZero.head)
    #dfWithoutZero.to_csv('C:/Users/Paris/Documents/ML/nutrients_csvfileNew.csv', index=False)
    dfWithoutZero.to_csv('C:/Users/Paris/Documents/ML/chatbot/nutritionNew.csv', index=False)

def findCategory(dflarge, dfSmall):
    for i, row in dflarge.iterrows():
        name_words = row['name'].split()
        for word in name_words:
            word1 = dfSmall['Food'].str.lower().replace('raw', '')
            word2 = word.lower().replace('raw', '')
            if any(word1 == word2):
                category = dfSmall.loc[word1 == word2, 'Category'].iloc[0]
                dflarge.loc[i, 'Category'] = category
                break  

    dfLarge.to_csv('C:/Users/Paris/Documents/ML/chatbot/nutritioCategory.csv', index=False)
    dflarge_with_category = dflarge[dflarge['Category'].notnull()]
    dflarge_with_category.to_csv('C:/Users/Paris/Documents/ML/chatbot/foodWithCategory.csv', index=False)

if __name__ == '__main__':
    dfSmall = pd.read_csv('C:/Users/Paris/Documents/ML/chatbot/nutrients_csvfile.csv')
    dfLarge = pd.read_csv('C:/Users/Paris/Documents/ML/chatbot/nutrition.csv')

    removeZero(dfSmall)
    removeZero(dfLarge)

    dfSmall = pd.read_csv('nutrients_csvfileNew.csv')
    dfLarge = pd.read_csv('nutritionNew.csv')

    #find food's category based on small dataset
    findCategory(dfLarge, dfSmall)
   
    # Find minimum per column
    min_values = dfLarge.min()

    # Find maximum per column
    max_values = dfLarge.max()

    print("Minimum values per column:")
    for value in min_values.values:
        print(value)
    print("\nMaximum values per column:")
    for value in max_values.values:
        print(value)
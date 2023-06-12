'''
    Links used:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
        https://www.geeksforgeeks.org/how-to-rename-multiple-column-headers-in-a-pandas-dataframe/
        https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
        https://www.geeksforgeeks.org/python-pandas-series-str-get_dummies/
        https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
        https://www.includehelp.com/python/change-a-column-of-yes-or-no-to-1-or-0-in-a-pandas-dataframe.aspx

    Dataset Link:
        https://www.kaggle.com/datasets/nithilaa/fitness-analysis?resource=download

'''

import pandas as pd


def exploreData(df, colFlag=False):
    
    print(df.head(), '\n')
    print(df.info(), '\n')
    print(df.describe(include = 'all').transpose(), '\n')
    print(df.isnull().sum(), '\n')
    
    if colFlag:
        for col in df.columns:
            print(col,'\n', df[col].dtype,'\n', df[col].unique())
    print('\n')


def mapColValues(df, cols_map_dod, printFlag=False):
    for col in cols_map_dod.keys():
        df[col] = df[col].map(cols_map_dod[col])
        if printFlag:
            print(col,'\n', df[col].dtype,'\n', df[col].unique())



def main():
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
 
    csvName = 'fitness analysis.csv'
    df = pd.read_csv(csvName)
    #exploreData(df)#; print(df.columns)

    # Drop columns
    df.drop(["Timestamp", "Your name ",], axis=1, inplace=True)

    renameCols_dict = {"Your gender ":"gender", "Your age ": "age_group", "How important is exercise to you ?": "ex_imp", "How do you describe your current level of fitness ?": "fitness_level",
                       "How often do you exercise?": "ex_freq", "What barriers, if any, prevent you from exercising more regularly?           (Please select all that apply)": "ex_barr_",
                       "What form(s) of exercise do you currently participate in ?                        (Please select all that apply)": "ex_form_", "Do you exercise ___________ ?":"do_ex_",
                       "What time if the day do you prefer to exercise?": "ex_time", "How long do you spend exercising per day ?":"ex_duration", "Would you say you eat a healthy balanced diet ?":"bal_diet_",
                       "What prevents you from eating a healthy balanced diet, If any?                         (Please select all that apply)":"diet_barr_", "How healthy do you consider yourself?": "health_level",
                       "Have you ever recommended your friends to follow a fitness routine?":"recommend_workout", "Have you ever purchased a fitness equipment?": "fitness_equip", 
                       "What motivates you to exercise?         (Please select all that applies )":"motiv_" 
                       }
    df.rename(columns=renameCols_dict, inplace=True)
    
    # Remove outliers from 'motiv_' column
    valuesToRemove_list = ['My dad motivates me ',  'Going to class on time 😉', 'Reaching class on time 😅','See the answer to what barriers, if any, prevent you from exercising  regularly ']  
    df = df[~df['motiv_'].isin(valuesToRemove_list)]


    # Map 'simple' categorical (nominal and scalar) columns
    cols_map_dod = {"gender":{"Female":0, "Male":1}, "fitness_equip": {"No":0, "Yes":1}, "recommend_workout": {"No":0, "Yes":1},
                    'fitness_level': {'Unfit':0, 'Average':1, 'Good':2, 'Very good':3, 'Perfect':4 },
                    'age_group': {'15 to 18':0, '19 to 25':1, '26 to 30':2, '30 to 40':3, '40 and above':4},
                    'ex_freq': {'Never':0, '1 to 2 times a week':1, '2 to 3 times a week':2, '3 to 4 times a week':3, '5 to 6 times a week':4, 'Everyday':5},
                    'ex_time':{'Early morning':0,  'Afternoon':2,'Evening':3},
                    'ex_duration':{"I don't really exercise":0, '30 minutes':1, '1 hour':2, '2 hours':3, '3 hours and above':4},
                    }
    mapColValues(df, cols_map_dod)

    # Categorical columns of strings to dummies  
    cols_string_dummies_dict = { "ex_barr_":";", "ex_form_":";", "do_ex_":"",  "bal_diet_":"", "diet_barr_":";", "motiv_":";"}
    for col in cols_string_dummies_dict.keys():

        if cols_string_dummies_dict[col]:
            df_dummies = df[col].str.get_dummies(cols_string_dummies_dict[col])
        else:
            df_dummies = df[col].str.get_dummies()
        #print(type(df_dummies))
        for dum_col in  df_dummies.columns:
            df_dummies = df_dummies.rename(columns={dum_col: col+dum_col})
        df = pd.concat([df, df_dummies], axis=1)   
        df.drop([col], axis=1, inplace=True)
    
    #print(df.head());  print(df.columns)
    
    # Rename properly dummie columns
    renameCols_dict = {'ex_barr_Allergies':'ex_barr_allergies', 
                       "ex_barr_I am not regular in anything":'ex_barr_notRegularPerson', 
                       "ex_barr_I can't stay motivated":"ex_barr_notMotivated", "ex_barr_I exercise regularly with no barriers":"ex_barr_noBarr",
                       "ex_barr_Less stamina":"ex_barr_lessStam", 
                       "ex_barr_My friends don't come ": "ex_barr_noFriend", 
                       "ex_barr_No gym near me":"ex_barr_noGym", "ex_barr_I don't really enjoy exercising":"ex_barr_noEnjoy",
                       "ex_barr_I have an injury": "ex_barr_injury",
                       
                        'ex_form_Gym':'ex_form_gym',"ex_form_I don't really exercise":'ex_form_noEx', 'ex_form_Lifting weights':"ex_form_weights",
                        'ex_form_Swimming':'ex_form_swim', 'ex_form_Team sport':'ex_form_team', 'ex_form_Walking or jogging': "ex_form_walk_jog", 
                        'ex_form_Yoga':'ex_form_yoga', 'ex_form_Zumba dance':'ex_form_zumba',
                        
                        'do_ex_Alone': 'do_ex_alone', "do_ex_I don't really exercise":'do_ex_noEx', 'do_ex_With a friend':'do_ex_friend',
                        'do_ex_With a group':'do_ex_group', 'do_ex_Within a class environment':'do_ex_env',
                        
                        'bal_diet_No':'bal_diet_no', 'bal_diet_Not always':'bal_diet_notAlways', 'bal_diet_Yes':'bal_diet_yes',

                        'diet_barr_Alcohol does me a good diet': 'diet_barr_alcohol', 'diet_barr_Cost':'diet_barr_cost', 'diet_barr_Lack of time':'diet_barr_noTime',
                        'diet_barr_Social circle':'diet_barr_social', "diet_barr_I do not measure. I can't say for sure if my diet is balanced. ":'diet_barr_notSure',
                        "diet_barr_i don't have a proper diet":"diet_barr_noDiet",

                        'motiv_Personal reasons':'motiv_personnal',
                        'motiv_I want to achieve a sporting goal':'motiv_sporting_goal',
                        'motiv_I want to increase muscle mass and strength': 'motiv_strength',
                        'motiv_I want to lose weight':'motiv_lose_weight',
                        'motiv_I want to relieve stress':'motiv_stress_out',
                        'motiv_I want to be flexible':'motiv_flexible',

                        }
    df.rename(columns=renameCols_dict, inplace=True) #; print(df.head()); print(df.columns)

    # Merge dummie columns of similar meaning
    groupsOR_dol = {'ex_barr_lazy': ['ex_barr_I am lazy', "ex_barr_I'll become too tired", 'ex_barr_Laziness', 'ex_barr_Laziness ',"ex_barr_I'm too lazy",'ex_barr_Laziness mostly ', 'ex_barr_Lazy'],
                    'ex_barr_travel':['ex_barr_Travel', 'ex_barr_Travel time I skip'],  
                    'ex_barr_busy':['ex_barr_I always busy with my regular works', "ex_barr_I don't have enough time"],

                    'diet_barr_junk':['diet_barr_Ease of access to fast food', 'diet_barr_Temptation and cravings'],
                    'diet_barr_noBarr':['diet_barr_Rarely eat fast food',  'diet_barr_I have a balanced diet'],  
                    
                    
                    'motiv_healthy': ['motiv_I dont wanna gain weight😉','motiv_I want to think clearly and I want to play cricket with my grandkids','motiv_Control Diabetes ',
                                      'motiv_Gotta get that alcohol and **** out of the system', 'motiv_Doing exercises prevents many diseases.So yeah saves a lot of money .',
                                      'motiv_Exercising gives you discipline and focus and removed bad thoughts from your mind.', 'motiv_To maintain healthy body and mind'],
                    'motiv_fit':['motiv_I want to be fit','motiv_I want to look young and think young'],
                    'motiv_noMotiv':["motiv_I'm sorry ... I'm not really interested in exercising", 'motiv_Not doing exercise'],
        }                            
       
    for col in groupsOR_dol.keys():
        df[col] = df[ groupsOR_dol[col]].any(axis=1).astype(int)
        df.drop( groupsOR_dol[col] , axis=1, inplace=True)

    print(df.head());  print(df.columns)    
    print(df.shape)

    df.to_csv('fitness.csv')    


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
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:55:39 2023

@author: Fik
"""

'''
    Links used:
        https://www.digitalocean.com/community/tutorials/python-remove-spaces-from-string


    Datasets' Link:
        https://www.kaggle.com/datasets/nithilaa/fitness-analysis?resource=download
        https://www.kaggle.com/datasets/niharika41298/gym-exercise-data

'''

import pandas as pd
import string

def exploreData(df, colFlag=False):
    
    print(df.head(), '\n')
    print(df.info(), '\n')
    print(df.describe(include = 'all').transpose(), '\n')
    print(df.isnull().sum(), '\n')
    
    if colFlag:
        for col in df.columns:
            print(col,'\n', df[col].dtype,'\n', df[col].unique())
    print('\n')


def dropUnnamed(df):
    for col in df.columns:
        if  'Unnamed' in col:
            df.drop([col], axis=1, inplace=True)


def getUserInfo(a_rec, printMotivs_flag=False):   
    
    motivColNames_list = ['motiv_personnal', 'motiv_sporting_goal', 'motiv_strength', 'motiv_lose_weight', 'motiv_stress_out', 'motiv_flexible', 'motiv_noMotiv']                     
    motivs_list = []       
    
    for col in motivColNames_list:
        if a_rec[col]:
            motivs_list.append(col.replace('motiv_', ''))
    
    if printMotivs_flag:
        print(motivs_list)
    
    fitness_level =  a_rec['fitness_level']
    ex_freq =  a_rec['ex_freq']
    ex_duration =  a_rec['ex_duration']
    
    return motivs_list, fitness_level, ex_freq, ex_duration 
    

def propose_exercises(df, motivs_list, fitness_level, ex_freq, ex_duration, bmi):
    '''
    Type ['Strength' 'Plyometrics' 'Stretching' 'Powerlifting' 'Strongman' 'Cardio' 'Olympic Weightlifting']
    BodyPart ['Abdominals' 'Abductors' 'Adductors' 'Biceps' Calves' 'Chest' 'Forearms' 'Glutes' 'Hamstrings' 'Lats' 'Lower Back' 'Middle Back' 'Traps' 'Quadriceps' 'Shoulders' 'Triceps']
    Equipment ['Bands' 'Barbell' 'Kettlebells' 'Dumbbell' 'Other' 'Cable' 'Machine' 'Body Only' 'Medicine Ball' 'None' 'Exercise Ball' 'Foam Roll' 'E-Z Curl Bar']
    '''
    
    if len(motivs_list)==1 and motivs_list[0]=='noMotiv':
        print("Go have sex multiple times, it's a very good exercise and you don't need motiv to have sex.\nIf you can't find someone to have sex with,\ndon't be a malakas and find a motiv.")
        #print('I know it is difficult to find a motiv. However, you should consider your health in the long term, you could at least try some stretcing')
        motivs_list.append('flexible')
        equipment_flag = 'no'
    
    elif len(motivs_list)==1 and motivs_list[0]=='sporting_goal':
        print('Since your only motivation is to achieve a sporting goal you should consult a personnal trainer.')
        return ''
    
    else:
        while(True):
            equipment_flag = input("Would you like to use equipement to your training (a simple yes or no is sufficient)? ").lower().translate({ord(c): None for c in string.whitespace})
            if equipment_flag=='yes' or equipment_flag=='no':
                break
            else:
                print("Sorry I don't understand your answer...\n")
    
    # Remove unecessary type of exercising
    df = df[df['Type'] != 'Olympic Weightlifting']
    
    # Decide about difficulty level
    if fitness_level==0 or fitness_level==1:
        df = df[df['Level'] == 0]
    elif fitness_level==2 or fitness_level==3:
        df = df[df['Level'] == 1]
    else:
        df = df[df['Level'] == 2]
    
    # Decide about equipment
    if equipment_flag=='yes':
        df = df[(df['Equipment'] != 'None') & (df['Equipment'] != 'Body Only')]
    else: 
        df = df[(df['Equipment'] == 'None') | (df['Equipment'] != 'Body Only') ]
    
    # Keep only types of exercise relative to user's motivs
    type_exerc_set = set() #is to be list
    motivs_types_dict = {'personnal': ['Strength'], 'strength': ['Powerlifting', 'Strongman'], 'lose_weight': ['Plyometrics'], 'stress_out':['Cardio'], 'flexible':['Strength']}
    for motiv in motivs_types_dict.keys() :
        type_exerc_set.update( motivs_types_dict[motiv] + ['Stretching'])
    
    # Remove types of exercise if they are too advanced for user's fitness level
    setToCheck = {}
    if fitness_level<3:
        setToCheck = {'Powerlifting', 'Strongman'}
    if fitness_level<2:
        setToCheck.add('Plyometrics')
    
    intersection = type_exerc_set.intersection(setToCheck)
    if intersection:
        type_exerc_set = type_exerc_set - intersection 
        type_exerc_set.add('Strength')
        
        
    type_exerc_list = list(type_exerc_set); print(type_exerc_list)
    df = df[df['Type'].isin(type_exerc_list)]
    
    df = df.sort_values(by=['Type', 'Rating'], ascending=[True, False])
    print(df.head())
    
    if  ex_freq and ex_duration: # if they are both different than 0
        number_exerc = (ex_freq+1) * ex_duration * 5 #  is to change based on Hippo's answer  
    else:
        number_exerc = 2 * 1 * 5
        
            
def main():
    #configure the display.max.columns option to make sure pandas doesn’t hide any columns
    pd.set_option("display.max.columns", None)
 

    df_fit = pd.read_csv('fitness.csv')
    dropUnnamed(df_fit)#; print(df_fit.head())
    
    a_rec = df_fit.iloc[1]# a_rec: a user's record
    #print(type(a_rec)) # -> <class 'pandas.core.series.Series'>
    
    '''
    'fitness_level': {'Unfit':0, 'Average':1, 'Good':2, 'Very good':3, 'Perfect':4 },
    'ex_freq': {'Never':0, '1 to 2 times a week':1, '2 to 3 times a week':2, '3 to 4 times a week':3, '5 to 6 times a week':4, 'Everyday':5},
    'ex_duration':{"I don't really exercise":0, '30 minutes':1, '1 hour':2, '2 hours':3, '3 hours and above':4},
    '''

    motivs_list, fitness_level, ex_freq, ex_duration = getUserInfo(a_rec)
    print(motivs_list, fitness_level, ex_freq, ex_duration, '\n\n')

    assumedBMI = 19.8 # https://www.calculator.net/bmi-calculator.html?ctype=metric&cage=23&csex=f&cheightfeet=5&cheightinch=10&cpound=160&cheightmeter=169.5&ckg=57&printit=0&x=56&y=14
    
    s = ''
    if ex_freq>=4 and ex_duration >=3: # 5-6, or 7 times a week, >=3h
        s+= "WOW! Keep up the good work, but you should consider the possibility you exercise too much.\n"\
            "Perhaps you sould give yourself some rest and take a dayor two off.\nAlternatively you could reduce"\
            " a little bit of your exercise time.\n"
    elif ex_freq==3 and ex_duration >=3:  # 3-4 times a week, 2h
        s += 'Congratulations! You are in good shape, keep up champ!\n'
    elif ex_freq>=1 and ex_duration >=1:
        #s += "Your fitness level is not that bad. Let's see how I can challenge you:\n"
        if ex_freq==3 and ex_duration==1: # 3-4 times a week, 30' => 90'- 120' ~ 1,5h - 2h
                s+= "Your fitness level is not that bad. Let's see how I can challenge you:\n"\
                    "Perhaps you could try to increase your exercise time per 30' everytime you train.\n"\
                    "Do not try to increase your exercise time too much.\nAlternatively, you could add a training or two of 30'.\n"
        elif (ex_freq==3 and ex_duration==2) or ex_duration==3:  
            # 3-4 times a week, 1h => 3h - 4h 
            # ex_freq==1 : 1-2 times a week, 2h => 2h - 4h 
            # ex_freq==2: 2-3 times a week, 2h => 4h - 6h
            s+= "Congratulations! You are in good shape, keep up champ!\n"\
                "If you want an extra challenge, you could try to increase your exercise time per 30' everytime you train.\n"\
                "Do not try to increase your exercise time too much.\nAlternatively, you could add a training or two of 30' or a training of 1h.\n"
        if ex_freq!=3 and ex_duration!=3:
            s+= "Perhaps you could try to increase your exercise time per 30' everytime you train.\n"\
                "Alternatively, you could try to increase the times you train per week by adding an extra training.\n"\
                "Do not try both changes simultaniously.\nDo not try to increase your exercise time too much or to increase"\
                " the times you train more than one per week.\nIf you try so, you might end up fatigue your self, ending up"\
                " quiting and having oposite results.\nIt is important to challenge yourself one step at a time and create steady habits.\n"
    else:
        if fitness_level>1:
            s += "Perhaps your fitness level is not as good as you think.\n"
            fitness_level = 0
        else:
            s += "Your fitness level is a little low\n"
        s+= "However we are here to fix that, let's help you improve it!\nFor starters you could try to exercise 1 to 2 times"\
            " per weak, at least per 30' each time.\nTry to create a steady exercise habbit to begin with before taking any"\
            " further actions to ameliorate your fitness level and your health level.\n"
    
    print(s)
    
    if ex_freq<3 and ex_duration<3:
        #df_gym = pd.read_csv('gym_no_dummies.csv')  
        df_gym = pd.read_csv('gym_no_dummies_noPowerStrongmanOW.csv')
        dropUnnamed(df_gym )
        print('\n', df_gym.head(),'\n')
        propose_exercises(df_gym, motivs_list, fitness_level, ex_freq, ex_duration, assumedBMI)



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
    
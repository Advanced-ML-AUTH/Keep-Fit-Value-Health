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
    

def getYesOrNo(prompt_string):

    while(True):
        flag = input(prompt_string).lower().translate({ord(c): None for c in string.whitespace})
        if flag=='yes':
            flag = True
            break
        elif flag=='no':
            flag = False
            break
        else:
            print("Sorry I don't understand your answer...\n")
    return flag


def decideEquip(df):
    equipment_flag = getYesOrNo("Would you like to use equipement to your training (a simple yes or no is sufficient)? ")
    # Decide about equipment
    equip_list = ['None','Body Only']
    if equipment_flag: # equipment_flag=='yes'
        df = df[~df['Equipment'].isin(equip_list)]
    else: 
        df = df[df['Equipment'].isin(equip_list)]
        
    return df


def decideDiffLevel(df, fitness_level):
    
    # Decide about difficulty level
    if fitness_level==0 or fitness_level==1:
        df = df[df['Level'] == 0]
    elif fitness_level==2 or fitness_level==3:
        df = df[df['Level'] == 1]
    else:
        df = df[df['Level'] == 2]
    return df


def decideTypeOnMotiv(df, motivs_list, fitness_level, bmi):
    # Keep only types of exercise relative to user's motivs
    type_exerc_set = set() #is to be list
    motivs_types_dict = {'personnal': ['Strength'], 'strength': ['Strength', 'Cardio'], 'lose_weight': ['Plyometrics'], 'stress_out':['Cardio', 'Stretching'], 'flexible':['Strength', 'Stretching']}
    for motiv in motivs_list:
        type_exerc_set.update(motivs_types_dict[motiv])  
    
    #print(type_exerc_set ) # to comment out
    
    # Remove types of exercise if they are too advanced for user's fitness level
    if fitness_level<2:
        if 'Plyometrics' in type_exerc_set:
            type_exerc_set.remove('Plyometrics')
            type_exerc_set.add('Strength')
    
    if bmi!= 0 and bmi!=1:# to check: 'Insufficient_Weight':0, 'Normal_Weight':1
        type_exerc_set.add('Cardio')
    
    type_exerc_list = list(type_exerc_set)#; print(type_exerc_list)
    
    df = df[df['Type'].isin(type_exerc_list)]
          
    return df
   
 
def decideMuscleGroupExercises(df):
    muscle_groups_dict = {'Upper Body':['Chest', 'Shoulders'], 'Arms':['Triceps', 'Biceps', 'Forearms'],
                          'Back':['Lats', 'Middle Back','Lower Back','Traps'], 
                          'Lower Body':['Quadriceps','Hamstrings','Glutes', 'Calves'], 
                          'Core':['Abdominals','Abductors', 'Adductors']
                          }
    #'''
    muscle_groups_flag_dict = {'Upper Body':False, 'Arms':False, 'Back':False, 'Lower Body':False, 'Core':False}
    
    # neck & shoulders
    flag = getYesOrNo("Do you experience any neck or shoulder pain (a simple yes or no is sufficient)?")
    if not flag:
        flag = getYesOrNo("Would you like any neck or shoulder exercises (a simple yes or no is sufficient)?")
    muscle_groups_flag_dict['Upper Body']=flag
    
    # arms
    flag = getYesOrNo("Would you like any arm exercises (a simple yes or no is sufficient)?")

    muscle_groups_flag_dict['Arms']=flag
    
    # back & core
    flag = getYesOrNo("Do you experience any back or core pain (a simple yes or no is sufficient)?")
    if not flag:
        flag = getYesOrNo("Would you like any back or core exercises (a simple yes or no is sufficient)?")
    muscle_groups_flag_dict['Back']=flag
    muscle_groups_flag_dict['Core']=flag

    # hips & legs
    flag = getYesOrNo("Do you experience any hip or leg pain (a simple yes or no is sufficient)?")
    if not flag:
        flag = getYesOrNo("Would you like any hip or leg exercises (a simple yes or no is sufficient)?")
    muscle_groups_flag_dict['Lower Body']=flag
    
    #print(muscle_groups_flag_dict)
    '''
    
    muscle_groups_flag_dict = {'Upper Body':True, 'Arms':False, 'Back':True, 'Lower Body':True, 'Core':True} # to comment out
    #'''
    
    muscles_to_keep_list = []
    for group in muscle_groups_flag_dict.keys():

        if muscle_groups_flag_dict[group]: # keep group muscle that are true
            muscles_to_keep_list +=  muscle_groups_dict[group]      
    df = df[df['BodyPart'].isin(muscles_to_keep_list)]
    
    return df  
        

def decideNumExerc(ex_freq, ex_duration):
    
    avg_num_ex = 4 # ang number of exercises per 30'
    if  ex_freq and ex_duration: # if they are both different than 0
        number_exerc = (ex_freq+2) * ex_duration * avg_num_ex 
    else:
        number_exerc = 2 * 1 * avg_num_ex
    
    return number_exerc


def chooseSpecificExercises(df, motivs_list, num_exerc):
    
    muscle_to_group_dict = {'Chest':'Upper Body', 'Shoulders':'Upper Body', 'Triceps':'Arms', 'Biceps':'Arms', 'Forearms':'Arms', 'Lats':'Back', 
                            'Middle Back':'Back','Lower Back':'Back', 'Traps':'Back', 'Quadriceps':'Lower Body', 'Hamstrings':'Lower Body',
                            'Glutes':'Lower Body', 'Calves':'Lower Body', 'Abdominals':'Core','Abductors':'Core', 'Adductors':'Core'
                           }    
    df_exercises =pd.DataFrame(columns = list(df.columns) )
    grouped_df = df.groupby(['Type', 'BodyPart'])
    
    
    while (len(df_exercises) < num_exerc): # at least as many exercises as to cover the exercise time     
        for group, group_data in grouped_df: # at least one exercise from each group-combination of Type and BodyPart
            col1_value, col2_value = group
            #print(f"Group: {col1_value}, {col2_value}"); print('len(group_data): ',len(group_data))
            df_exercises.loc[len(df_exercises)] = group_data.iloc[0] 
            group_data = group_data.iloc[1:]
    
    df_exercises['MuscleGroup']= df_exercises['BodyPart'].map(muscle_to_group_dict)
    
    df_exercises = df_exercises.sort_values(by=['MuscleGroup', 'Type', 'BodyPart','Rating'], ascending=[True, True, True, False]) 
    #print(df_exercises) # to comment out
    return df_exercises    
    

def presentExercises(df, fitness_level, ex_freq):
    no_equip_list = ['None','Body Only']
    ex_freq_dict = {0:"didn't exercised", 1:'exercised 1 to 2 times a week', 2:'exercised 2 to 3 times a week', 3:'exercised 3 to 4 times a week'}
    
    s = "After taking into consideration your BMI, your personal habits, your needs and the fitness level you provided,"\
        "I propose the following exercises:\n"
    
    for i in range(len(df)):
        # Access row data using df.loc[i, 'column_name'] or df.iloc[i, index]
        s+= ('Exercise ' + str(i+1) +': '+ df.loc[i, 'Title']+', how to:\n\n'+df.loc[i, 'Desc']+'.\n\nIt focuses on your '+
             df.loc[i, 'MuscleGroup'].lower()+' muscles, especially your '+ df.loc[i, 'BodyPart']+'.\n')
        if df.loc[i, 'Equipment'] not in no_equip_list: 
            s+= 'Equipment you will need:'+ df.loc[i, 'Equipment']+'\n\n'
        else:
            s+='\n\n'
    
    s+= 'Some general guidelines:\nKeep in mind you should try to exercise at least for '
    if fitness_level<2:
        s+="150' per week, ideally for 30' per training minimum (~ 5 trainings per week on average).\n" 
    else:
        s+="75' per week, ideally for 20' per training minimum (~ 3 trainings per week on average).\n" 
    s+= ("However, until now you "+ ex_freq_dict[ex_freq]+".\nIf you don't meet the minimum advised training time per week you should try"+
         " to increase your trainings to "+ex_freq_dict[(ex_freq+1)].replace("exercised ", "")+".\nAlternatively, you could try to increase"\
         " your training time per 30' each time you train. It would be better of you did not try both changes simultaniously.\nDo not "\
         "try to increase your exercise time too much or to increase the times you train more than one per week.\nIf you try so,"\
         " you might end up fatigue your self, ending up quiting and having oposite results.\nIt is important to challenge yourself"\
         " one step at a time and create steady habits.\n"
         )
    
    
    print(s)
    
    
def proposeExercises(df, motivs_list, fitness_level, ex_freq, ex_duration, bmi):
    '''
    Type ['Strength' 'Plyometrics' 'Stretching' 'Cardio']
    BodyPart ['Abdominals' 'Abductors' 'Adductors' 'Biceps' Calves' 'Chest' 'Forearms' 'Glutes' 'Hamstrings' 'Lats' 'Lower Back' 'Middle Back' 'Traps' 'Quadriceps' 'Shoulders' 'Triceps']
    '''
    
    if len(motivs_list)==1 and motivs_list[0]=='noMotiv': # Check if user has noMotiv at all
        print("I know it is difficult to find a motiv. However, you should consider your health in the long term, you could at least try some stretcing")
        motivs_list.append('flexible')    
    elif len(motivs_list)==1 and motivs_list[0]=='sporting_goal': # Check if users only motiv is a sporting goal
        print('Since your only motivation is to achieve a sporting goal you should consult a personnal trainer.')
        return ''
    
    if 'noMotiv' in motivs_list:
        motivs_list.remove('noMotiv')
    #print(motivs_list) # to comment out
    #print('df', len(df))  
    
    # Ask user about equipment and remove df rows respectively
    df = decideEquip(df) #; print('df', len(df))
    
    # Decide physical exercises' difficulty level, based on user's fitness level
    df = decideDiffLevel(df, fitness_level) #; print('df', len(df))
    
    # Keep only rows for which 'Type' column is aligned with user's motivation, also considering his fitness level and his BMI
    df = decideTypeOnMotiv(df, motivs_list, fitness_level, bmi)  #; print('df', len(df))
    
    # Keep only rows for which 'BodyPart' column is aligned with "special needs" of user's muscle groups
    df = decideMuscleGroupExercises(df) #; print('df', len(df))
    
    #df = df.sort_values(by=['Type', 'Rating'], ascending=[True, False]) ; print(df.head(), '\n\n')    
    num_exerc = decideNumExerc(ex_freq, ex_duration); print(num_exerc)
    
    df_exercises = chooseSpecificExercises(df, motivs_list, num_exerc)
    presentExercises(df_exercises, fitness_level, ex_freq)
    
    if 'sporting_goal'in motivs_list:
        print('If you want to set any sporting goal(s) you should consider to consult a personnal trainer.')  
        
            
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

    motivs_list, fitness_level, ex_freq, ex_duration = getUserInfo(a_rec) # !!!
    
    #print(motivs_list, fitness_level, ex_freq, ex_duration, '\n\n') # to comment out

    assumedBMI = 1 # check
        
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
        df_gym = pd.read_csv('gym_no_dummies_noPowerStrongmanOW.csv')
        dropUnnamed(df_gym ) #; print('\n', df_gym.head(),'\n')
        # for col in df_gym.columns: print(col,'\n', df_gym[col].dtype,'\n', df_gym[col].unique())
        proposeExercises(df_gym, motivs_list, fitness_level, ex_freq, ex_duration, assumedBMI)



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
    
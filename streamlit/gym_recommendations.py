import streamlit as st
import pandas as pd
import string

def getUserInfo():  
    fitness_level = getFitnsessLevel() 
    ex_freq = getExcFrequency()
    ex_duration = getExcDuration()
    motivs_list = getMotive()

    return motivs_list, fitness_level, ex_freq, ex_duration


def getMotive():
    # Define the options for the checkboxes
    options = ['Personal', 'Sporting goal', 'Strength', 'Lose weight', 'Relax', 'Flexibility', 'No motivation']

    # Render the checkboxes as a group
    selected_options = st.multiselect("What is your motivation to excercise?", options)

    ex_duration = {
        'Personal': 'personnal', 
        'Sporting goal': 'sporting_goal', 
        'Strength': 'strength', 
        'Lose weight': 'lose_weight', 
        'Relax': 'stress_out', 
        'Flexibility': 'flexible', 
        'No motivation': 'noMotiv'
        }
    
    durationList = []
    if selected_options:
        for option in selected_options:
            if ex_duration[option]:
                durationList.append(ex_duration[option])

        return durationList                


def getExcDuration():
    # Define the string values for the slider
    options = ['Select an option', 'I don\'t really exercise', '30 minutes', '1 hour', '2 hour', '3 hour and above']

    # Render the selectbox/slider
    selected_value = st.selectbox('How long do you exercise for?', options)

    ex_duration = {
        "I don't really exercise": '0', 
        '30 minutes': '1', 
        '1 hour': '2', 
        '2 hours': '3', 
        '3 hours and above': '4'
        }
    
    if selected_value in ex_duration:
        return ex_duration[selected_value]  

def getExcFrequency():
    # Define the string values for the slider
    options = ['Select an option', 'Never', '1 to 2 times a week', '2 to 3 times a week', '3 to 4 times a week', '4 to 5 times a week', '5 to 6 times a week', 'Everyday']

    # Render the selectbox/slider
    selected_value = st.selectbox('How frequent do you excercise?', options)

    ex_freq =  {
        'Never': '0', 
        '1 to 2 times a week': '1', 
        '2 to 3 times a week': '2', 
        '3 to 4 times a week': '3', 
        '5 to 6 times a week': '4', 
        'Everyday': '5'
        }
    
    if selected_value in ex_freq:
        return ex_freq[selected_value]

def getFitnsessLevel():
    # Define the string values for the slider
    options = ['Select an option', 'Unfit', 'Average', 'Good', 'Very Good', 'Perfect']

    # Render the selectbox/slider
    selected_value = st.selectbox('What\'s your fitness level?', options)

    fitness_level =  {
        'Unfit': '0', 
        'Average': '1', 
        'Good': '2', 
        'Very good': '3', 
        'Perfect': '4' 
        }
    
    if selected_value in fitness_level:
        return fitness_level[selected_value]
    

def dropUnnamed(df):
    for col in df.columns:
        if  'Unnamed' in col:
            df.drop([col], axis=1, inplace=True)


def getGymOutputText(ex_freq, ex_duration, fitness_level, motivs_list, bmi):
    s = ''
    if ex_freq>=4 and ex_duration >=3: # 5-6, or 7 times a week, >=3h
        s+= "WOW! Keep up the good work, but you should consider the possibility you exercise too much.\n\n"\
            "Perhaps you sould give yourself some rest and take a dayor two off.\n\nAlternatively you could reduce"\
            " a little bit of your exercise time.\n\n"
    elif ex_freq==3 and ex_duration >=3:  # 3-4 times a week, 2h
        s += 'Congratulations! You are in good shape, keep up champ!\n\n'
    elif ex_freq>=1 and ex_duration >=1:
        if ex_freq>3 or ex_duration>=1:
            s += "Your fitness level is not that bad. Let's see how I can challenge you:\n\n"
        elif ex_freq==3 and ex_duration==1: # 3-4 times a week, 30' => 90'- 120' ~ 1,5h - 2h
                s+= "Your fitness level is not that bad. Let's see how I can challenge you:\n\n"\
                    "Perhaps you could try to increase your exercise time per 30' everytime you train.\n\n"\
                    "Do not try to increase your exercise time too much.\nAlternatively, you could add a training or two of 30'.\n\n"
        elif (ex_freq==3 and ex_duration==2) or ex_duration==3:  
            # 3-4 times a week, 1h => 3h - 4h 
            # ex_freq==1 : 1-2 times a week, 2h => 2h - 4h 
            # ex_freq==2: 2-3 times a week, 2h => 4h - 6h
            s+= "Congratulations! You are in good shape, keep up champ!\n\n"\
                "If you want an extra challenge, you could try to increase your exercise time per 30' everytime you train.\n\n"\
                "Do not try to increase your exercise time too much.\n\nAlternatively, you could add a training or two of 30' or a training of 1h.\n\n"
    else:
        if fitness_level>1:
            s += "Perhaps your fitness level is not as good as you think.\n\n"
            fitness_level = 0
        else:
            s += "Your fitness level is a little low.\n\n"
        s+= "However we are here to fix that, let's help you improve it!\n\nFor starters you could try to exercise 1 to 2 times"\
            " per weak, at least per 30' each time.\n\nTry to create a steady exercise habbit to begin with before taking any"\
            " further actions to ameliorate your fitness level and your health level.\n\n"
    
    st.info(s)
    hasRecommendations = False
    if ex_freq<3 and ex_duration<3:
        df_gym = pd.read_csv('C:/Users/kyriaki.potamopoulou/Documents/DWS/ML/gym/recommendations/gym_no_dummies_noPowerStrongmanOW.csv')
        dropUnnamed(df_gym ) 
        hasRecommendations = proposeExercises(df_gym, motivs_list, fitness_level, ex_freq, ex_duration, bmi)

    return hasRecommendations


def proposeExercises(df, motivs_list, fitness_level, ex_freq, ex_duration, bmi):
    '''
    Type ['Strength' 'Plyometrics' 'Stretching' 'Cardio']
    BodyPart ['Abdominals' 'Abductors' 'Adductors' 'Biceps' Calves' 'Chest' 'Forearms' 'Glutes' 'Hamstrings' 'Lats' 'Lower Back' 'Middle Back' 'Traps' 'Quadriceps' 'Shoulders' 'Triceps']
    '''
    
    if len(motivs_list)==1 and motivs_list[0]=='noMotiv': # Check if user has noMotiv at all
        st.wrtie("I know it is difficult to find a motive. However, you should consider your health in the long term, you could at least try some stretcing")
        motivs_list.append('flexible')    
    elif len(motivs_list)==1 and motivs_list[0]=='sporting_goal': # Check if users only motiv is a sporting goal
        st.write('Since your only motivation is to achieve a sporting goal you should consult a personnal trainer.')
        return ''
    
    if 'noMotiv' in motivs_list:
        motivs_list.remove('noMotiv') 

    if 'sporting_goal'in motivs_list:
        st.warning('If you want to set any sporting goal(s) you should consider to consult a personnal trainer.')
        motivs_list.remove('sporting_goal')

    st.divider()

    respond = False; 
    # Ask user about equipment and remove df rows respectively
    df, equipment_flag = decideEquip(df) 
    
    # Decide physical exercises' difficulty level, based on user's fitness level
    if equipment_flag:
        df = decideDiffLevel(df, fitness_level) 
        
        # Keep only rows for which 'Type' column is aligned with user's motivation, also considering his fitness level and his BMI
        df = decideTypeOnMotiv(df, motivs_list, fitness_level, bmi)  
        
        # Keep only rows for which 'BodyPart' column is aligned with "special needs" of user's muscle groups
        df, hasResponded = decideMuscleGroupExercises(df) 
        
        if hasResponded == True:
            num_exerc = decideNumExerc(ex_freq, ex_duration)
            
            df_exercises = chooseSpecificExercises(df, motivs_list, num_exerc)

            st.divider()

            presentExercises(df_exercises, fitness_level, ex_freq, equipment_flag)
            
            respond = True  

    return respond      



def decideEquip(df):
    equipment_flag = getYesOrNo("Would you like to use equipement to your training? ")
    # Decide about equipment
    equip_list = ['None','Body Only']
    if equipment_flag: # equipment_flag=='yes'
        df = df[~df['Equipment'].isin(equip_list)]
    else: 
        df = df[df['Equipment'].isin(equip_list)]
    
    return df, equipment_flag


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
        
    # Remove types of exercise if they are too advanced for user's fitness level
    if fitness_level<2:
        if 'Plyometrics' in type_exerc_set:
            type_exerc_set.remove('Plyometrics')
            type_exerc_set.add('Strength')
    
    if bmi!= 0 and bmi!=1:# to check: 'Insufficient_Weight':0, 'Normal_Weight':1
        type_exerc_set.add('Cardio')
    
    type_exerc_list = list(type_exerc_set)#;
    
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
    flag, hasResponded1 = getYesOrNo("Do you experience any neck or shoulder pain?")

    if hasResponded1:
        if not flag:
            flag, hasResponded1 = getYesOrNo("Would you like any neck or shoulder exercises?")
      
        muscle_groups_flag_dict['Upper Body']=flag
        
        if hasResponded1:
            # arms
            flag, hasResponded3 = getYesOrNo("Would you like any arm exercises?")

            muscle_groups_flag_dict['Arms']=flag
            
            if hasResponded3:
                # back & core
                flag, hasResponded4 = getYesOrNo("Do you experience any back or core pain?")
                if not flag:
                        flag, hasResponded4 = getYesOrNo("Would you like any back or core exercises?")
                if hasResponded4:
                    muscle_groups_flag_dict['Back']=flag
                    muscle_groups_flag_dict['Core']=flag

                    # hips & legs
                    flag, hasResponded5 = getYesOrNo("Do you experience any hip or leg pain?")
                    if not flag:
                        flag, hasResponded5 = getYesOrNo("Would you like any hip or leg exercises?")
                    muscle_groups_flag_dict['Lower Body']=flag
                    
                    '''
                    muscle_groups_flag_dict = {'Upper Body':True, 'Arms':False, 'Back':True, 'Lower Body':True, 'Core':True} # to comment out
                    #'''
                    if hasResponded5:
                        muscles_to_keep_list = []
                        for group in muscle_groups_flag_dict.keys():

                            if muscle_groups_flag_dict[group]: # keep group muscle that are true
                                muscles_to_keep_list +=  muscle_groups_dict[group]      
                        df = df[df['BodyPart'].isin(muscles_to_keep_list)]
        
                        return df, hasResponded5

    return False, False

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
            df_exercises.loc[len(df_exercises)] = group_data.iloc[0] 
            group_data = group_data.iloc[1:]
    
    df_exercises['MuscleGroup']= df_exercises['BodyPart'].map(muscle_to_group_dict)
    
    df_exercises = df_exercises.sort_values(by=['MuscleGroup', 'Type', 'BodyPart','Rating'], ascending=[True, True, True, False]) 
    return df_exercises    
    

def presentExercises(df, fitness_level, ex_freq, equipment_flag):
    no_equip_list = ['None','Body Only']
    ex_freq_dict = {0:"didn't exercised", 1:'exercised 1 to 2 times a week', 2:'exercised 2 to 3 times a week', 3:'exercised 3 to 4 times a week'}
    
    st.info("After taking into consideration your BMI, your personal habits, your needs and the fitness level you provided,\n\n"\
        "I propose the following exercises:\n\n")
    
    for i in range(len(df)):
        # Access row data using df.loc[i, 'column_name'] or df.iloc[i, index]
        with st.expander('Exercise ' + str(i+1) +': '+ df.loc[i, 'Title']):
            st.write('How to:\n\n'+df.loc[i, 'Desc']+'.\n\nIt focuses on your '+
                df.loc[i, 'MuscleGroup'].lower()+' muscles, especially your '+ df.loc[i, 'BodyPart']+'.\n\n')
            if equipment_flag == True and df.loc[i, 'Equipment'] and  df.loc[i, 'Equipment'] not in no_equip_list: 
                equipment = df.loc[i, 'Equipment']
                st.write(f'Equipment you will need: {equipment} \n\n')

    st.divider()
        
    s =""
    s+= 'Some general guidelines:\n\nKeep in mind you should try to exercise at least for '
    if fitness_level<2:
        s+="150' per week, ideally for 30' per training minimum (~ 5 trainings per week on average).\n\n" 
    else:
        s+="75' per week, ideally for 20' per training minimum (~ 3 trainings per week on average).\n\n" 
    s+= ("However, until now you "+ ex_freq_dict[ex_freq]+".\n\nIf you don't meet the minimum advised training time per week you should try"+
         " to increase your trainings to "+ex_freq_dict[(ex_freq+1)].replace("exercised ", "")+".\n\nAlternatively, you could try to increase"\
         " your training time per 30' each time you train. It would be better of you did not try both changes simultaniously.\n\nDo not "\
         "try to increase your exercise time too much or to increase the times you train more than one per week.\n\nIf you try so,"\
         " you might end up fatigue your self, ending up quiting and having oposite results.\n\nIt is important to challenge yourself"\
         " one step at a time and create steady habits.\n\n"
         )
    
    
    st.warning(s)
    st.divider()

    

def getYesOrNo(prompt_string):
    # Define the string values for the slider
    options = ['Select an option', 'Yes', 'No']

    # Render the selectbox/slider
    selected_value = st.selectbox(prompt_string, options)

    options =  {
        'Yes': True,
        'No': False 
        }
    
    if selected_value in options:
        hasResponded = True
        return options[selected_value], hasResponded

    return False, False

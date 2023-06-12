import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import json
import joblib
import randomForest_bmi as bmi
import userInputs as userInputs
import nutritionRecomendations as nutritionRec
import gym_recommendations as gymRec

def classifyObesity(inputs):
    df = pd.DataFrame(inputs)
    df= bmi.scaleCols(df, std_colsToScale_list=['Height','Weight'], minMax_colsToScale_list= ['Age'])
    model = joblib.load('C:/Users/kyriaki.potamopoulou/Documents/DWS/ML/obesity.joblib')
    # Use the model for prediction
    prediction = model.predict(df)
    return prediction[0]


def userCharacteristics():
    gender = userInputs.getGender()
    age = userInputs.getAge()
    weight = userInputs.getWeight()
    height = userInputs.getHeight()
    tech = userInputs.get_technology()
    aggreeTerms = False

    if (gender and age and weight and height):
        changeLine()
        st.success("Thank you for providing me these useful info!")
        aggreeTerms = userInputs.show_terms_and_conditions()
        inputs = {
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'TUE': [int(tech)],
            'Gender': [gender]
        }	
        print(inputs)
    if(aggreeTerms == True):
        if 'bmi' not in st.session_state or st.session_state.bmi == None:
            st.session_state.bmi = classifyObesity(inputs)
            predictions = st.session_state.bmi
        else:
            predictions = st.session_state.bmi
        
        getHealthRecomendations(predictions, gender, age)
    else: 
        st.session_state.bmi = None


def getHealthRecomendations(predictions, gender, age):
    if predictions == 1:
        st.info("Awesome! You are in a good shape. However you can improve further either your food habits or your fitness level.")
        
    else:
        st.info("It seems that you can improve your food habits and your fitness levels")

    st.divider()

    input = {
        'gender': gender,
        'age': age,
        'bmi': predictions
    }
    chatbotNutrition("first", input, "")



def findMessage(type):
    if type == "first":
        message_mapping = {
        'yeah',
        'yes',
        'sure',
        'of course',
        'ok',
        'okay'
        }
    
    return message_mapping


def endFlow(type):
    if type == "first":
        message_mapping = {
        'no',
        'nope',
        'no sure',
        'not sure'
        }
    
    return message_mapping

def check_string_in_list(sentence, answer_list):
    for answer in answer_list:
        if sentence in answer:
            return True
    return False


def changeLine():
    st.markdown("<br>",unsafe_allow_html=True)
        
def recomendationOptions(input):
    option = userInputs.getRecomendationOption()
    st.divider()
    if option == '1':
        retrieveFoodRec(input, False)
    elif option == '0':
        retrieveGymRec(input, False)


def retrieveGymRec(input, isFromNutrition):
    motivs_list, fitness_level, ex_freq, ex_duration = gymRec.getUserInfo()
    if motivs_list and fitness_level and ex_freq and ex_duration:
        hasRecommendations = gymRec.getGymOutputText(int(ex_freq), int(ex_duration), int(fitness_level), motivs_list, input['bmi'])
        if hasRecommendations == True and not isFromNutrition:
            nutrtionOption = userInputs.getNutritionOption()
            if nutrtionOption == 'yes':
                retrieveFoodRec(input, True)
            elif nutrtionOption == 'no':
                st.info("Thank you for using this app, I'm hoping that I gave useful advice!\n\n Have a nice day 🎈")
        elif hasRecommendations == True and isFromNutrition:
            st.info("Thank you for using this app, I'm hoping that I gave useful advice!\n\n Have a nice day 🎈")
    

def retrieveFoodRec(input, isFromGym):
    nutritionModel = joblib.load('C:/Users/kyriaki.potamopoulou/Documents/DWS/ML/nutrition.joblib')

    col_output = nutritionRec.getNutritionFood(input)

    hasRecomendation = nutritionRec.getRecNutritionOutput(col_output, nutritionModel)
    if not isFromGym:
        gymOption = userInputs.getGymOption()
        if gymOption == 'yes':
            retrieveGymRec(input, True)
        elif gymOption == 'no':
            st.info("Thank you for using this app, I'm hoping that I gave useful advice!\n\n Have a nice day 🎈")
    elif hasRecomendation:
        st.info("Thank you for using this app, I'm hoping that I gave useful advice!\n\n Have a nice day 🎈")


def chatbotNutrition(count, input, response):    
    
    if count == 'first':
        firstMessage = st.text_input("Would you like to assist you with a plan that aims to your improvement?")

    elif count == 'second':
        firstMessage = st.text_input(f"Can you please be more specific in your answer? Would you like to assist you with a plan that aims to your improvement? {response}")

        
    if firstMessage and check_string_in_list(firstMessage.lower(), findMessage("first")):
        recomendationOptions(input)

    elif firstMessage and  check_string_in_list(firstMessage.lower(), endFlow("first")):
        response = generate_end_response()
        st.text_area("Chatbot Response:", value=response)

    elif firstMessage:
       response = "I can't understand your answer: " + firstMessage
       chatbotNutrition("second", input, response)



def chatbot(count, answer):    
    st.markdown("""<style>.stTextInput > label {
        font-size:100px; font-weight:bold;}</style>""",unsafe_allow_html=True) 
    
    if count == 'first':
        firstMessage = st.text_input("Are you looking for assistance?")
    
    elif count == 'second':
        firstMessage = st.text_input(f"Can you please be more specific in your answer? Are you looking for assistance? {answer}")

    if firstMessage and check_string_in_list(firstMessage.lower(), findMessage("first")):
        secondMessage ="Sure! Firstly, give some information about you to determine your body fat levels"
        response = generate_response(firstMessage, secondMessage)
        st.markdown(secondMessage)
        return True
        

    elif firstMessage and  check_string_in_list(firstMessage.lower(), endFlow("first")):
        response = generate_end_response()
        st.text_area("Chatbot Response:", value=response)

    elif firstMessage:
       response = "I can't understand your answer: " + firstMessage
       chatbot("second", response)

    return False

def generate_response(user_input, response):
    return response


def generate_end_response():
    response = "It's OK. Have a nice day!"
    return response


def main():
    st.set_page_config(page_title="Keep Fit Value Health", layout="wide")

    with open("C:/Users/kyriaki.potamopoulou/Documents/DWS/ML/data/fitnessGif.json", "r") as f:
        data = json.load(f)
    st_lottie(data, speed=0.3, height=200, key="initial")

    st.markdown(
                "<div align='left' style='padding-bottom: 50px;font-size: 24px;'><b>Keep Fit Value Health</b> App is an interactive mobile application designed to help users maintain a healthy lifestyle by providing personalized guidance on nutrition and exercise.<br>The app aims to enhance users' understanding of their food choices and gym habits, assisting them in making informed decisions to achieve their fitness goals.</div>",
                unsafe_allow_html=True
            )
    row1_spacer1, row1_1, row1_spacer2 = st.columns((0.7, 2.5, 0.7))

    with row1_1:

        st.subheader("Hello, I am a recommendation chatbot responsible for nutritional food and fitness topics!")
        positiveAnswer = chatbot("first", "")

        if positiveAnswer == True:
            userCharacteristics()


if __name__ == "__main__":
    main()
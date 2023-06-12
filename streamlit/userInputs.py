import streamlit as st
import re

def getGender():
    gender_options = ['Select an option', 'Female', 'Male',  "I'd rather not to say"]

    selected_option = st.selectbox('Select gender', gender_options)

    message_mapping = {
        'Female': '0',
        'Male': '1',
        'I\'d rather not to say': ''
    }

    if selected_option in message_mapping:
        return message_mapping[selected_option]
    

def getIsCooked():
    cooked_options = ['Cooked', 'Raw']

    selected_option = st.selectbox('Has the food been prepared/cooked?', cooked_options)

    message_mapping = {
        'Cooked': 'Yes',
        'Raw': 'No'
    }

    if selected_option in message_mapping:
        return message_mapping[selected_option]

def getAge():
    changeLine()

    ageMessage = st.text_input("Enter your age")
    age_match = re.search(r'\d+', ageMessage)

    if not age_match:
        return ""
    elif age_match:
        age = int(age_match.group())  # Convert the extracted string to an integer
        if age >= 0 and age <= 100:  # Assuming valid age range is between 0 and 100
            return age
        else: st.error("Please enter a valid age")

    return ""


def getWeight():
    changeLine()

    weightMessage = st.text_input("Enter your weight (kg)")
    weight_match = re.findall(r'\d+(?:[.,]\d+)?', weightMessage)

    for match in weight_match:
        if match:
            try:
                weight_value = float(match)  # Convert the extracted weight to a float
                if weight_value > 0.0 and weight_value <= 300.0:  # Assuming valid weight range is between 0 and 300
                    return weight_value
                else:
                    st.error("Please enter valid weight")
                    return ""
            except ValueError:
                st.error("Please enter valid weight")


def getHeight():
    changeLine()

    heightMessage = st.text_input("Enter your height (cm)")
    height_match = re.findall(r'\d+(?:[.,]\d+)?', heightMessage)

    for match in height_match:
        if match:
            try:
                value = float(match)  # Convert the extracted weight to a float
                if value > 100.0 and value <= 230.0:  # Assuming valid weight range is between 100 and 230 cm
                    return value/100
                else:
                    st.error("Please enter valid height")
                    return ""
            except ValueError:
                st.error("Please enter valid height")


def get_frequency(type):
    changeLine()

    if type == 'vegies':
        selected_option = st.slider("From 0-4 how frequent do you consume vegetables?", max_value=4)
    if type == 'meals':
        selected_option = st.slider("How many main meals do you consume?", max_value=4)
    if type == 'snacks':
        selected_option = st.slider("From 0-3 how frequent do you consume food between meals?", max_value=3)
    if type == 'water':
        selected_option = st.slider("From 0-3 how frequent do you drink water?", max_value=3)
    if type == 'activity':
        selected_option = st.slider("From 0-3 how frequent do you excersise?", max_value=3)
    if type == 'alcohol':
        selected_option = st.slider("From 0-3 how frequent do you consume alcohol?", max_value=3)


    if selected_option:
        return selected_option


def get_technology():
    changeLine()

    selected_option = st.slider("From 0-2 How often do you use technology devices?",min_value=0, max_value=2)
    if selected_option:
        return str(selected_option)

    return "0"


def get_booleans(type):
    changeLine()

    options = ['Select an option', 'Yes', 'No']

    if type == 'family_history':
        selected_option = st.selectbox('Do you have obesity in your family history?', options)
    if type == 'high_calories':
        selected_option = st.selectbox('Do you usualy consume high caloric foods?', options)
    if type == 'smoke':
        selected_option = st.selectbox('Do you smoke?', options)
    if type == 'monitor_calories':
        selected_option = st.selectbox('Do you monitor calories consumption?', options)

    option_mapping = {
        'No': '0',
        'Yes': '1'
    }

    if selected_option in option_mapping:
        return option_mapping[selected_option]


def get_transportation():
    changeLine()

    options = ['Select an option', 'automobile', 'bike', 'motorbike', 'walking']

    selected_option = st.selectbox('What is your mean of transportation?', options)
    
    if selected_option == 'automobile':
        return '1', '0', '0', '1', '0'
    elif selected_option == 'bike':
        return '0', '1', '0', '1', '0'
    elif selected_option == 'motorbike':
        return '0','0', '1', '1', '0'
    elif selected_option == 'walking':
        return '0', '0', '0', '0', '1'
    else:
        return '', '','', '',''


def changeLine():
    st.markdown("<br>",unsafe_allow_html=True)

def show_terms_and_conditions():
    changeLine()

    with st.expander("Terms & Conditions"):
        st.write("""
        We are committed to providing personalized nutritional and fitness recommendations to enhance your well-being. This consent statement outlines how we collect and utilize your information.
        1. Personal Information Collection:
        The chatbot may collect personal information such as your age, gender, weight, height, dietary preferences, and fitness goals. This information is voluntary and will be used to provide tailored recommendations.
        2. Nutrition and Fitness Recommendations:
        Based on the information provided, the chatbot will offer suggestions and guidance regarding nutritional food choices and fitness exercises. These recommendations are general in nature and should not replace professional medical advice.
        3. Information Protection:
        We take the security and confidentiality of your personal information seriously. We implement appropriate measures to safeguard it from unauthorized access, disclosure, alteration, or destruction.
        4. Data Usage:
        The personal information collected by the chatbot will be used solely for the purpose of providing nutrition and fitness recommendations. We may analyze aggregated and anonymized data for research and improvement purposes.
        5. Information Sharing:
        We do not sell, trade, or transfer your personal information to third parties without your explicit consent unless required by law.
        6. Consent Withdrawal:
        You have the right to withdraw your consent for the collection and usage of your personal information.
        You can discontinue using the chatbot or contact us to request the deletion of your data.
        7. Accuracy of Information:
        Please ensure that the personal information you provide is accurate and up-to-date. It is your responsibility to update us if any changes occur.
        By using this chatbot, you acknowledge that you have read and understood this consent statement and agree to its terms.
        """)
    
    st.caption("By using this chatbot, you hereby consent to the collection and use of your personal information as described in this statement.")
    agree = st.checkbox("I agree to the Terms and Conditions")
    if agree:
        return True

def getRecomendationOption():
    changeLine()
    gender_options = ['Select an option', 'Gym', 'Nutrition']

    selected_option = st.selectbox('Do you prefer assistance for nutritional food or fitness?', gender_options)

    message_mapping = {
        'Gym': '0',
        'Nutrition': '1'
    }

    if selected_option in message_mapping:
        return message_mapping[selected_option]

def getNutritionOption():
    changeLine()
    gender_options = ['Select an option', 'Yes', 'No']

    selected_option = st.selectbox('Do you need assistance for nutritional food?', gender_options)

    message_mapping = {
        'Yes': 'yes',
        'No': 'no'
    }

    if selected_option in message_mapping:
        return message_mapping[selected_option]

def getGymOption():
    changeLine()
    gender_options = ['Select an option', 'Yes', 'No']

    selected_option = st.selectbox('Do you need assistance for gym excercises?', gender_options)

    message_mapping = {
        'Yes': 'yes',
        'No': 'no'
    }

    if selected_option in message_mapping:
        return message_mapping[selected_option]
    

def getNumOfFoods():
    changeLine()

    message = st.text_input("Enter the number of different food items you ate today.")
    match = re.search(r'\d+(?:[.,]\d+)?', message)

    if not match:
        return ""
    elif match:
        value = int(match.group())  # Convert the extracted string to an integer
        if value >= 0 and value <= 20:  
            return value
        else: st.error("Please enter a valid number")

    return ""
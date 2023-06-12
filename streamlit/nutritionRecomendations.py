from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np 
import pandas as pd
import streamlit as st
import userInputs as userInputs
import re

def generate_bert_embeddings(food_names):
    # Load BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')

    # Tokenize food names
    food_tokens = food_names.apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True, padding='max_length'))

    # Convert tokens to PyTorch tensors
    food_tensors = [torch.tensor([token_ids]) for token_ids in food_tokens]

    # Obtain BERT embeddings
    with torch.no_grad():
        food_embeddings = [model(token_ids)[0].mean(dim=1).squeeze().numpy() for token_ids in food_tensors]
    food_embeddings = np.array(food_embeddings)

    return food_embeddings


def get_user_selected_categories(user_inputs, trained_model, scaler, category_labels,pca):
    categories = []
    food_with_category = pd.DataFrame(columns=['food', 'category'])  
    drinks_calories = 0
    total_calories = 0
    total_sugars = 0
    st.subheader("The food categories that consume today are:\n\n")
    # Loop through the rows of the dataframe
    for index, row in user_inputs.iterrows():
        name = row['name']
        serving_size = row['serving_size']
        calories = row['calories']
        total_fat = row['total_fat']
        protein = row['protein']
        carbohydrate = row['carbohydrate']
        fiber = row['fiber']
        sugars = row['sugars']
        
        # Create an array with the input values
        input_values = np.array([[serving_size, calories, total_fat, protein, carbohydrate, fiber, sugars]])
               
        # Generate BERT embeddings for the food name
        food_name_embeddings = generate_bert_embeddings(pd.Series([name]))
        
        # Concatenate the food name embeddings with the scaled input values
        input_features = np.concatenate([food_name_embeddings, input_values], axis=1)
        
        # Scale the input values using the same scaler used during training
        scaled_input_values = scaler.transform(input_features)
        
        scaled_input_values_pca = pca.transform(scaled_input_values)
        
        # Make predictions using the trained model
        predicted_categories = trained_model.predict(scaled_input_values_pca)
        
        # Get the category label based on the predicted numerical label
        category_label = category_labels[predicted_categories[0]]
        if category_label == 'Drinks,Alcohol, Beverages':            
            drinks_calories = drinks_calories + calories
            
        total_sugars = total_sugars + sugars
        total_calories = total_calories + calories
        
        showCategories(name, category_label)
        
        categories.append(category_label)
        
        # Add row to the DataFrame
        food_with_category = pd.concat([food_with_category, pd.DataFrame({'food': name, 'category': category_label}, index=[0])], ignore_index=True)
        
    if total_calories != 0:
        drink_calories_perc = drinks_calories/total_calories
    else:
        drink_calories_perc = 0
        
    return categories, food_with_category, drink_calories_perc, total_sugars

# Print the predicted category label for the current food item
def showCategories(name, category_label):
    if category_label == 'Fruits':
        st.write(f"- **{name}**, Predicted Category: 🍒 {category_label}")

    elif category_label == 'Vegetables':
        st.markdown(f"- **{name}**, Predicted Category: 🥗 {category_label}")
    
    elif category_label == 'Breads, cereals, fastfood,grains':
        st.write(f"- **{name}**, Predicted Category: 🥖 {category_label}")
    
    elif category_label == 'Dairy products':
        st.write(f"- **{name}**, Predicted Category: 🧀 {category_label}")
    
    elif category_label == 'Meat Poultry':
        st.write(f"- **{name}**, Predicted Category: 🍗 {category_label}")
    
    elif category_label == 'Desserts':
        st.write(f"- **{name}**, Predicted Category: 🍨 {category_label}")
    
    elif category_label == 'Fish Seafood':
        st.write(f"- **{name}**, Predicted Category: 🦞 {category_label}")
    
    elif category_label == 'Fats Oils Shortenings':
        st.write(f"- **{name}**, Predicted Category: 🏺 {category_label}")
    
    elif category_label == 'Drinks,Alcohol, Beverages':
        st.write(f"- **{name}**, Predicted Category: 🍹 {category_label}")

    else:
        st.write(f"- **{name}**, Predicted Category: 🍒 {category_label}")
    

def count_categories(category_labels, categories):
    category_counts = {}
    
    # Initialize the dictionary with category labels as keys and count as 0
    for label in category_labels:
        category_counts[label] = 0
    
    # Count the occurrences of each category label
    for category in categories:
        if category in category_counts:
            category_counts[category] += 1
    
    return category_counts


def getThresholdFruits(age, sex):
    if age < 19:
        return 2.5  # 250gr /day
    elif age >= 19 and sex == 'Female':
        return 2.5  # 250 gr /day     
    elif 19 <= age < 60 and sex == 'Male':
        return 3.2  # 320 gr /day
    else:
        return 2.5

 
def getThresholdVegetables(age, sex, cooked_vegetables = True):
    # Determine the appropriate threshold for daily vegetables product consumption based on age
    if cooked_vegetables == 'Yes':
        if sex == 'Female':
            return 4  # 4 servings of 100gr each
        else: #male
            if age < 14 or age >= 60:
                return 4.5 # 4.5 servings of 100gr each
            elif 14 <= age <= 59:
                return 5 # 5 servings of 100gr each
    else: # raw leafy salad greens            
        if sex == 'Female':
            return 8  # 8 servings of 100gr each
        else: #male
            if age < 14 or age >= 60:
                return 9 # 9 servings of 100gr each
            elif 14 <= age <= 59:
                return 10 # 10 servings of 100gr each    


def getThresholdGrains(age):
    # Determine the appropriate threshold for daily grains product consumption based on age
    if age < 9:
        return 1.4  # 140gr /day
    else:
        return 2.25  # 225 gr /day
    

def getThresholdDairy():
    # Determine the appropriate threshold for daily grains product consumption based on age
    return 4
    

def getThresholdMeat():
    return 1.7
    

def getThresholdDesert(sex):
    if sex == 'Male':
        return 0.4 # 0.4 servings (1 serving = 100gr) 
    else:
        return 0.25


def getThresholdFish():
    return 0.5


def getThresholdFat(age, sex):
    if age <= 13 and age >30 and sex == 'Female':
        return 0.9  
    elif (14 <= age <=30  and sex == 'Female') or (9 <= age <= 13  and sex == 'Male'):
        return 1.15
    elif age>14 and sex == 'Male':
        return 1.4



# Toddlers and children of 2-3 years old are not taken into account, since they should not be able to use the app
def getRecomendation(category_counts, input, drink_calories_perc, total_sugars):
    # recommendation - fruits
    age_series = pd.to_numeric(input['age'], errors='coerce')
    age = int(age_series.iloc[0])
    sex = input['sex'].iloc[0]

    #threshold of raw vegetables is twice the threshold of cooked vegetables
    vegetableIsCooked = input.loc[input['category'] == 'Vegetables']
    if not vegetableIsCooked.empty:
        for vegetable in vegetableIsCooked.iterrows():
            if vegetable[1]['isCooked'] == 'Yes':
                category_counts['Vegetables'] +=1

    threshold_drinks = 0.15    

    if category_counts['Fruits'] ==  getThresholdFruits(age, sex):
        st.write('🍒 Good job! You have reached the recommended servings of fruits per day!')
    elif category_counts['Fruits'] >  getThresholdFruits(age, sex):
        st.write('🍒 You seem to be a fruit lover! Don\'t forget to also include other food categories in your daily diet!')
    elif category_counts['Fruits'] <  getThresholdFruits(age, sex) and  getThresholdFruits(age, sex) != 0:
        st.markdown('🍒 Great job on eating fruits! They are very important for a balanced diet. The recommended servings of fruit per day is 5 servings of 100g each.')
    elif  getThresholdFruits(age, sex) == 0:
        st.write('🍒 Fruits are an excellent source of vitamins, minerals, and dietary fiber. They contribute to overall health and well-being, and you should try including them in your daily diet.')
        st.write('I would recommend eating 5 servings of fruits per day.')

    st.divider()

    # recommendation - vegetables
    if category_counts['Vegetables'] == getThresholdVegetables(age, sex):
        st.write("🥗 Well done! You have reached the recommended servings of vegetables per day!")
    elif category_counts['Vegetables'] > getThresholdVegetables(age, sex):
        st.write("🥗 You're doing a great job with your vegetable intake! Keep up the good work and also don't forget to include other food categories in your daily diet!")
    elif category_counts['Vegetables'] < getThresholdVegetables(age, sex):
        st.write("🥗 Vegetables are an essential part of a healthy diet. Aim to include more vegetables in your daily meals. The recommended servings of vegetables per day is 3-5.")
    
    st.divider()

    # recommendation - grains
    if category_counts['Breads, cereals, fastfood,grains'] == getThresholdGrains(age):
        st.write("🥖 Fantastic! You have reached the recommended servings of grains per day!")
    elif category_counts['Breads, cereals, fastfood,grains'] > getThresholdGrains(age):
        st.write("🥖 You're doing well with your grain consumption. You should consider reducing them a little bit and balance it with other food groups.")
    elif category_counts['Breads, cereals, fastfood,grains'] < getThresholdGrains(age):
        st.write("🥖 Grains are an important source of energy and nutrients. Aim to include more whole grains in your diet. The recommended servings of grains per day is 6-8 servings of 150g each.")
    
    st.divider()

    # recommendation - dairy
    if category_counts['Dairy products'] == getThresholdDairy():
        st.write("🧀 Great job! You have reached the recommended servings of dairy products per day!")
    elif category_counts['Dairy products'] > getThresholdDairy():
        st.write("🧀 You're doing well with your dairy intake, but it might be beneficial to reduce it slightly and diversify your nutrient sources.")
    elif category_counts['Dairy products'] < getThresholdDairy():
        st.write("🧀 Dairy products are a good source of calcium and other essential nutrients. Aim to include more dairy products in your daily diet. The recommended amount of dairy products per day is 200gr-300gr.")
    
    st.divider()

    # recommendation - meat and poultry
    if category_counts['Meat Poultry'] == getThresholdMeat():
        st.write("🍗 Well done! You have reached the recommended servings of meat and poultry per day!")
    elif category_counts['Meat Poultry'] > getThresholdMeat():
        st.write("🍗 You're doing well with your meat and poultry consumption. Remember to choose lean sources, vary your protein sources and balance them with other food groups.")
    elif category_counts['Meat Poultry'] < getThresholdMeat():
        st.write("🍗 Meat and poultry are important sources of protein and other nutrients. Aim to include more lean meat and poultry in your diet. The recommended servings of meat and poultry per day is 3 servings of 100g each.")
    
    st.divider()

    # recommendation - desserts
    if category_counts['Desserts'] > 0 and total_sugars < getThresholdDesert(sex):
        st.write("🍨 It's important to limit your consumption of desserts, as they often contain added sugars and unhealthy fats. Try to choose healthier dessert options or enjoy them in moderation.")
    elif category_counts['Desserts'] > 0 and total_sugars >= getThresholdDesert(sex):
        st.write("🍨 It seems like you enjoy desserts! However, it's important to consume them in moderation.")
        st.write("🍨 Consider reducing your dessert intake and opting for healthier alternatives like fresh fruits or yogurt.")
        st.write("🍨 Remember, desserts should be enjoyed as an occasional treat rather than a regular part of your daily diet.")
    
    st.divider()

    # recommendation - fish
    if category_counts['Fish Seafood'] == getThresholdFish():
        st.write("🦞 Good job! You have reached the recommended servings of fish per day! When it comes to fish consumption, the general recommendation is to aim for 2-3 servings per week.")
    elif category_counts['Fish Seafood'] > getThresholdFish():
        st.write("🦞 You're doing well with your fish consumption. Fish is a great source of omega-3 fatty acids. When it comes to fish consumption, the general recommendation is to aim for 2-3 servings per week.")
    elif category_counts['Fish Seafood'] < getThresholdFish():
        st.write("🦞 Fish is a nutritious food that provides omega-3 fatty acids. Aim to include more fish in your diet. The general recommendation is to aim for 2-3 servings per week.")
    
    st.divider()

    # recommendation - fats and oils
    if sex == "Male" and category_counts['Fats Oils Shortenings'] <= getThresholdFat(age, sex):
        st.write("🏺 Good job! You have reached the recommended daily intake of fats and oils for males.")
    elif sex == "Female" and category_counts['Fats Oils Shortenings'] <= getThresholdFat(age, sex):
        st.write("🏺 Good job! You have reached the recommended daily intake of fats and oils for females.")
    else:
        st.write("🏺 It's important to consume fats and oils in moderation. Aim to choose healthier sources of fats, such as olive oil, avocado, and nuts.")

    st.divider()

    # recommendation - drinks
    if drink_calories_perc >= threshold_drinks:
        st.write("🍹 You seem to drink a lot of drinks and beverages.")
        st.write("🍹 Try to choose beverages that are calorie-free—especially water——or that contribute beneficial nutrients, such as fat-free and low-fat milk and 100% juice")
        st.write("🍹 Coffee, tea, and flavored waters also are options, but the most nutrient-dense options for these beverages include little, if any, sweeteners or cream.")
    elif category_counts['Drinks,Alcohol, Beverages'] == 0:
        st.write("🍹 It seems that you haven't consumed any drinks today. ")
        st.write("🍹Try to choose beverages that are calorie-free—especially water——or that contribute beneficial nutrients, such as fat-free and low-fat milk and 100% juice")
    
    st.divider()

    # recommendation - soups   
    if category_counts['Soups'] > 0:
        st.write("🥣 When prepared with the right ingredients, soup can be a truly healthy dish with multiple nutritional benefits.")
        st.write("🥣 For example, soups made with bone-, vegetable-, or meat-based broths provide vitamins, minerals, and nutrients, such as collagen.")
        st.write("🥣 Be careful though! Not all soups are healthy. Some ingredients used to improve texture or taste like heavy cream may produce an unhealthy dish that you might want to eat less frequently.")


def getRecNutritionOutput(col_output, nutritionModel):
    if col_output.button('Output'): 
        user_inputs = st.session_state.df
        model = nutritionModel[0]
        category_labels = nutritionModel[1]
        scaler = nutritionModel[2]
        pca = nutritionModel[3]

        user_categories, food_with_category, drink_calories_perc, total_sugars = get_user_selected_categories(user_inputs,model, scaler,category_labels, pca)
        category_counts = count_categories(category_labels, user_categories)
        combined_inptus = pd.concat([food_with_category, user_inputs], axis=1)

        with st.expander("See nutrition recomendations"):
            getRecomendation(category_counts, combined_inptus,  drink_calories_perc, total_sugars)
        
        return True

    return False


def getCorrectValue(servingSize, value):
    value_match = re.findall(r'\d+(?:[.,]\d+)?', value)

    for match in value_match:
        if match:
            try:
                valueFloat = float(match) 
                return valueFloat * (servingSize / 100)
            except ValueError:
                st.error("Please enter valid value")


def getNutritionFood(input):

    age = input['age']
    if input['gender'] == '0':
        gender = "Female"
    else:
        gender = "Male"

    df = pd.DataFrame(columns=['name', 'serving_size', 'calories', 'total_fat', 'protein', 'carbohydrate', 'fiber', 'sugars'])

    if 'df' not in st.session_state:
        st.session_state.df = df
    else:
        df = st.session_state.df 


    col_name, col_cooked = st.columns(2)
    with col_name:
        name = st.text_input("Food")
    with col_cooked:
        isCooked = userInputs.getIsCooked()

    serving_size = st.number_input("Serving size")
    calories = getCorrectValue(serving_size, st.text_input("Calories"))
    total_fat = getCorrectValue(serving_size, st.text_input("Total fat"))
    protein = getCorrectValue(serving_size, st.text_input("Protein"))
    carbohydrate = getCorrectValue(serving_size, st.text_input("Carbohydrate"))
    fiber = getCorrectValue(serving_size, st.text_input("Fiber"))
    sugars = getCorrectValue(serving_size, st.text_input("Sugars"))

    col_add, col_remove, col_output = st.columns(3)

    # Add user input to DataFrame
    if col_add.button('Add'):
        new_row = {
            'age': str(age),
            'sex': gender,
            'name': name, 
            'serving_size': serving_size, 
            'calories': calories,
            'total_fat': total_fat,
            'protein': protein,
            'carbohydrate': carbohydrate,
            'fiber': fiber,
            'sugars': sugars,
            'isCooked': isCooked
            }
        
        df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)

        st.session_state.df = df

    if col_remove.button('Remove') and len( st.session_state.df ) > 0:
            df = st.session_state.df
            df = df.drop(0).reset_index(drop=True)
            st.session_state.df = df
            st.success('Row removed successfully!')

    st.table(df)

    return col_output
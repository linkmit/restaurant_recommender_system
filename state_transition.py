# for Linux operation systems

# !sudo apt install espeak
# !sudo apt install espeak-ng

import numpy as np
import pandas as pd
import Levenshtein
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
import pyttsx3
import time
from machine_learning import *
import globals

restaurant_df = pd.read_csv('restaurant_info_1c.csv', sep=';')
restaurant_df.head()

# apply rules to determine consequents of new properties from 1C
restaurant_df['assigned seats'] = restaurant_df['crowdedness'] == True
restaurant_df['children'] = restaurant_df['lengthofstay'] == False

def label_touristic(row):
    """ handle contradictions by prioritising food quality and price range over romanian.
    returns 2 if there is a contradction """
    global restaurant_df
    if (row['foodquality'] == True) & (row['pricerange'] == 'cheap'):
        if row['food'] == 'romanian':
            return 2  # indicate contradiction
        else:
            return 1
    return 0

def label_romantic(row):
    """ handle contradictions by prioritising crowdedness over length of state.
    returns 2 if there is a contradction """
    global restaurant_df
    if row['crowdedness'] == False:
        if row['lengthofstay'] == True:
            return 2  # indicate contradiction
        else:
            return 1
    return 0

restaurant_df['touristic'] = restaurant_df.apply(label_touristic, axis=1)
restaurant_df['romantic'] = restaurant_df.apply(label_romantic, axis=1)

# variable for storing all restaurant options
options= {}
for col in ['food', 'area', 'pricerange']:
  options[col] = restaurant_df[col].dropna().unique().tolist()

# list storing all additional options
additional_options=list(['romantic', 'touristic', 'assigned seats', 'children'])

# parent class for storing states
class DialogState:
    # initialises the system utterance and restaurant info as a dictionary with user preferences.
    def __init__(self, utterance=None, restaurant_info=None):
        self.utterance = utterance
        self.restaurant_info = restaurant_info or{
            'area': None,
            'food': None,
            'pricerange': None,
            'additionalrequirements': [],
            'restaurant': None
        }

    # handles input based on the state the system is in
    def handle_input(self, input):
        raise NotImplementedError()

# initial state that welcomes the user
class GreetingState(DialogState):
    def __init__(self):
        super().__init__("Hello, welcome to the Cambridge restaurant system! You can ask for restaurants by area, food type, or price range. How may I help you?")

    def handle_input(self, input):
        dialog_act = getDialogAct(input)

        # always end the conversation when dialog act is 'goodbye'
        if dialog_act == 'goodbye':
            return GoodbyeState()

        # update restaurant_info
        self.restaurant_info = handle_user_input_pref(input, self.restaurant_info)

        # check which info is still missing
        # if all preferences are filled in, go to ConfirmState
        if self.restaurant_info['area'] is None:
            return RequestInfoState(self.restaurant_info, 'area')
        elif self.restaurant_info['food'] is None:
            return RequestInfoState(self.restaurant_info, 'food')
        elif self.restaurant_info['pricerange'] is None:
            return RequestInfoState(self.restaurant_info, 'pricerange')
        else:
            return ConfirmState(self.restaurant_info)

# state that asks the user for more info on preferences
class RequestInfoState(DialogState):
    def __init__(self, restaurant_info=None, missing_info=None):
        super().__init__()
        self.restaurant_info = restaurant_info or {}
        if missing_info == 'reqalt':
            self.utterance = f"Resetting your preferences... \n What area, price range, and/or food are you looking for?"
        elif missing_info == 'nomatch':
            self.utterance = f"There were no restaurants matching your preferences. \n Resetting your preferences... \n What area, price range, and/or food are you looking for?"
        else:
            self.utterance = f"What {missing_info} are you looking for?"

    def handle_input(self, input):
        dialog_act = getDialogAct(input)
        if dialog_act == 'goodbye':
            return GoodbyeState()

        #update restaurant_info
        self.restaurant_info = handle_user_input_pref(input, self.restaurant_info)

        #check which info is still missing
        if not self.restaurant_info['area']:
            return RequestInfoState(self.restaurant_info, 'area')
        elif not self.restaurant_info['food']:
            return RequestInfoState(self.restaurant_info, 'food')
        elif not self.restaurant_info['pricerange']:
            return RequestInfoState(self.restaurant_info, 'pricerange')
        else:
            return ConfirmState(self.restaurant_info)

# state that asks user if the system understood preferences correctly
class ConfirmState(DialogState):
    def __init__(self, restaurant_info):
        super().__init__()
        self.restaurant_info = restaurant_info
        self.utterance = f"You are looking for a {restaurant_info['pricerange']} restaurant in the {restaurant_info['area']} area that serves {restaurant_info['food']} food. Is that correct?"

    # navigate to states based on what user says
    def handle_input(self, input):
        dialog_act = getDialogAct(input)

        if dialog_act == 'affirm':
            restaurant = None
            restaurant = getRestaurant(self.restaurant_info)

            if restaurant == None:
                self.restaurant_info = {
                    'area': None,
                    'food': None,
                    'pricerange': None,
                    'additionalrequirements': [],
                    'restaurant': None
                }
                return RequestInfoState(self.restaurant_info, 'nomatch')

            else:
                return AdditionalRequirementsState(self.restaurant_info)

        elif dialog_act == 'negate':
            self.restaurant_info = {
                    'area': None,
                    'food': None,
                    'pricerange': None,
                    'additionalrequirements': [],
                    'restaurant': None
                }
            return RequestInfoState(self.restaurant_info, 'reqalt')

        elif dialog_act == 'goodbye':
            return GoodbyeState()

        else:
            return self

# state that asks the user for additional requirements after basic preferences
class AdditionalRequirementsState(DialogState):
    def __init__(self, restaurant_info):
        super().__init__("Do you have additional requirements?")
        self.restaurant_info = restaurant_info

    def handle_input(self, input):
        dialog_act = getDialogAct(input)
        if dialog_act == 'goodbye':
            return GoodbyeState()
        if dialog_act == 'negate':
            return RecommendState(self.restaurant_info)

        self.restaurant_info = handle_user_input_addreq(input, self.restaurant_info)

        return RecommendState(self.restaurant_info)

# if all preferences have been confirmed, recommend restaurant
class RecommendState(DialogState):
    def __init__(self, restaurant_info):
        super().__init__()
        self.restaurant_info = restaurant_info
        self.restaurant_info['restaurant'] = getRestaurant(self.restaurant_info)
        # if there are no restaurants based on food, area, and price range, communicate that there are no matches.
        if self.restaurant_info['restaurant'] is None:
            self.utterance = f'There are no restaurants matching your basic requirements.'
        # if there is no match satisfying all additional requirements, still recommend a place.
        else:
            self.utterance = f"I recommend {self.restaurant_info['restaurant']}. {generate_extra_properties_utterance(self.restaurant_info)} \n Type 'info' for more information about the restaurant. \n Type 'another restaurant' to restart the search. \n Otherwise, type 'goodbye' to end the chat."

    # recommend based on dialog act of user
    def handle_input(self, input):
        input = input.lower()
        # ask for more info if user is negating
        if input == 'info':
            return WhichInfoState(self.restaurant_info)
        elif input == 'another restaurant':
            self.restaurant_info = {
                    'area': None,
                    'food': None,
                    'pricerange': None,
                    'additionalrequirements': [],
                    'restaurant': None
                }
            return RequestInfoState(self.restaurant_info)
        elif input == 'goodbye':
            return GoodbyeState()
        else:
            return self

class WhichInfoState(DialogState):
    def __init__(self, restaurant_info):
        super().__init__()
        self.restaurant_info = restaurant_info
        self.utterance = f"Type 'phone' to get the restaurant's phone number or 'address' to get the restaurant's address. \n"

    def handle_input(self, input):
        input = input.lower()
        # ask for more info if user is negating
        if input == 'phone':
            return InfoPhoneState(self.restaurant_info)

        elif input == 'address':
            return InfoAddressState(self.restaurant_info)

        elif input == 'goodbye':
            return GoodbyeState()

        else:
            return RecommendState(self.restaurant_info)

class InfoPhoneState(DialogState):
    def __init__(self, restaurant_info):
        super().__init__()
        self.restaurant_info = restaurant_info
        restaurant = restaurant_info['restaurant']
        phonenumber = get_info(restaurant, 'phone')
        if phonenumber:
            self.utterance = f"{restaurant}'s phone number is {phonenumber}. \n Type 'address' for the restaurant's location. \n Type 'another restaurant' to restart the search. \n Otherwise, type 'goodbye' to end the chat."
        else:
            self.utterance = f"Sorry, {restaurant}'s phone number is unavailable. Type 'address' for the restaurant's location. \n Type 'another restaurant' to restart the search. \n Otherwise, type 'goodbye' to end the chat."

    def handle_input(self, input):
        input = input.lower()
        # ask for more info if user is negating

        if input == 'address':
            return InfoAddressState(self.restaurant_info)

        elif input == 'another restaurant':
            self.restaurant_info = {
                    'area': None,
                    'food': None,
                    'pricerange': None,
                    'additionalrequirements': [],
                    'restaurant': None
                }
            return RequestInfoState(self.restaurant_info)

        elif input == 'goodbye':
            return GoodbyeState()

        else:
            return RecommendState(self.restaurant_info)

class InfoAddressState(DialogState):
    def __init__(self, restaurant_info):
        super().__init__()
        self.restaurant_info = restaurant_info
        restaurant = restaurant_info['restaurant']
        address = get_info(restaurant, 'address')
        if address:
            self.utterance = f"{restaurant} is located on {address}. Type 'phone' for the restaurant's phone number. \n Type 'another restaurant' to restart the search. \n Otherwise, type 'goodbye' to end the chat."
        else:
            self.utterance = f"Sorry, {restaurant}'s address is unavailable. Type 'phone' for the restaurant's phone number. \n Type 'another restaurant' to restart the search. \n Otherwise, type 'goodbye' to end the chat."

    def handle_input(self, input):
        input = input.lower()

        if input == 'phone':
            return InfoPhoneState(self.restaurant_info)

        # ask for more info if user is negating
        if input == 'another restaurant':
            self.restaurant_info = {
                    'area': None,
                    'food': None,
                    'pricerange': None,
                    'additionalrequirements': [],
                    'restaurant': None
                }
            return RequestInfoState(self.restaurant_info)

        elif input == 'goodbye':
            return GoodbyeState()

        else:
            return RecommendState(self.restaurant_info)

class GoodbyeState(DialogState):
    def __init__(self):
        super().__init__()
        self.utterance = "Goodbye! Enjoy your meal!"
        return self

def handle_user_input_pref(input, restaurant_info):
    global options
    split_input = input.split()  # Split input into a list of words

    # Look for exact matches to the list of options
    for word in split_input:
        for key, values in options.items():
            if word in values:
                restaurant_info[key] = word # Fill in key if there is a direct match
                split_input.remove(word) # Remove word from the list to prevent being matched again

    # For unfilled domains, check for close matches
    for domain, value in restaurant_info.items():
        if value is None and domain != 'restaurant' and domain !='additionalrequirements':
            min_distance = float('inf')
            closest_match = None

            # Measure levenshtein distances for each input word and available options in the domains
            for word in split_input:
                for option in options[domain]:
                    distance = Levenshtein.distance(word, option)

                    # Update min_distance and closest_match if it is smaller than the current distance
                    if distance < min_distance:
                        min_distance = distance
                        closest_match = option

            # Fill in domain with the closest match (within 3 Levenshtein distances)
            if closest_match and min_distance <= globals.threshold:
                restaurant_info[domain] = closest_match

    return restaurant_info

def handle_user_input_addreq(input, restaurant_info):
    global additional_options
    additional_options = additional_options.copy()
    user_input = input.lower()
    split_input = input.split()

    # Look for exact matches to the list of additional options
    for additional_option in additional_options:
        if additional_option in user_input:
            restaurant_info['additionalrequirements'].append(additional_option)
            user_input = user_input.replace(additional_option, '')
            split_input = [word for word in split_input if additional_option not in word]

    # For unfilled domains, check for close matches
    min_distance = float('inf')
    closest_match = None

    for word in split_input:
            for option in additional_options:
                distance = Levenshtein.distance(word, option)
                # Update min_distance and closest_match if it is smaller than the current distance
                if distance < min_distance:
                        min_distance = distance
                        closest_match = option

            # Fill in domain with the closest match (within 3 Levenshtein distances)
            if closest_match and min_distance <= globals.threshold:
                restaurant_info['additionalrequirements'].append(closest_match)

    return restaurant_info

def getDialogAct(input):
    # Use the decision tree classifier to classify dialog acts
    global vectorizer, clf_logreg
    input_transformed = vectorizer.transform([input])
    prediction = clf_logreg.predict(input_transformed)
    return prediction[0]

def getRestaurant(restaurant_info):
    global restaurant_df
    # Make a copy of the restaurant dataframe to filter through
    filtered_restaurants = restaurant_df.copy()
    # Filter by food type (if specified)
    if restaurant_info['food'] != None:
        filtered_restaurants = filtered_restaurants[filtered_restaurants['food'] == restaurant_info['food']]

    # Filter by area (if specified)
    if restaurant_info['area'] != None:
        filtered_restaurants = filtered_restaurants[filtered_restaurants['area'] == restaurant_info['area']]

    # Filter by price range (if specified)
    if restaurant_info ['pricerange'] != None:
        filtered_restaurants = filtered_restaurants[filtered_restaurants['pricerange'] == restaurant_info['pricerange']]

    if filtered_restaurants.empty:
        return None

    # Return the first restaurant in the filtered list
    return filtered_restaurants.iloc[0]['restaurantname']

def generate_extra_properties_utterance(restaurant_info):
    global restaurant_df

    # if the user specified no additional requirements, there are no additional explanations
    if not restaurant_info['additionalrequirements']:
        return " "
    
    reqs = restaurant_info['additionalrequirements'].copy()
    count = len(reqs)

    row = restaurant_df[restaurant_df['restaurantname'] == restaurant_info['restaurant']]
    restaurant_data = row[reqs]

    # generate descriptions for each additional requirements that are fulfilled.
    description = "The restaurant "
    # 1: true
    # 2: true but contradictory
    if 'romantic' in restaurant_info['additionalrequirements']:
        if restaurant_data['romantic'].values== 2:
            description += "is romantic because it is not crowded, but only allows short stays."
            count -= 1
            reqs.remove('romantic')
        elif restaurant_data['romantic'].values == 1:
            description += "is romantic because it is not crowded."
            count -= 1
            reqs.remove('romantic')

    if 'assigned seats' in restaurant_info['additionalrequirements']:
        if restaurant_data['assigned seats'].values> 0:
            description += " has assigned seats because it is very busy."
            count -= 1
            reqs.remove('assigned seats')

    if 'children' in restaurant_info['additionalrequirements']:
        if restaurant_data['children'].values > 0:
            description += " is great for children because the stay is short."
            count -= 1
            reqs.remove('children')

    if 'touristic' in restaurant_info['additionalrequirements']:
        if restaurant_data['touristic'].values == 1:
            description += " is touristic because the prices are low and the food is good."
            count -= 1
            reqs.remove('touristic')
        elif restaurant_data['touristic'].values == 2:
            description += " is touristic because the prices are low and the food is good. However, it is Romanian, which is not a popular choice for tourists."
            count -= 1
            reqs.remove('touristic')
            
    # if none of the additional requirements are fulfilled
    if count == len(restaurant_info['additionalrequirements']):
        return "However, we could not satisfy your additional requirements."

    # if some of the additional requirements are fulfilled, but others are not
    if count != 0:
        remaining_reqs = ', '.join(reqs[:-1]) + (" and " + reqs[-1] if len(reqs) > 1 else reqs[0])
        return description + f"However, we could not satisfy the rest of your additional requirements for {remaining_reqs}."

    # if all are fulfilled
    return description

def get_info(restaurant, info):
    df_copy = restaurant_df.copy()
    row = df_copy[df_copy['restaurantname'] == restaurant]

    if not row.empty:
        if info == 'phone':
            phonenumber = row['phone'].values[0]
            if pd.isna(phonenumber):
                return None
            return str(phonenumber)

        elif info == 'address':
            address = row['addr'].values[0]
            if pd.isna(address):
                return None
            return str(address)
    else:
        return None

def extra_config():
    # Levenshtein edit switch
    while True:
        levenshtein_edit_y_n = input("Would you like to edit the Levenshtein distance? [y/n] ").lower()
        if levenshtein_edit_y_n == 'y':
            while True:
                try:
                    globals.threshold = int(input("What should the distance be? Type a number. "))
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
            break
        elif levenshtein_edit_y_n == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # TTS switch
    while True:
        tts_enable_y_n = input("Would you like to enable text-to-speech?[y/n]")
        if tts_enable_y_n == 'y':
            globals.tts_enable = True
            break
        elif tts_enable_y_n == 'n':
            globals.tts_enable = False
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # caps lock switch
    while True:
        caps_lock_enable_y_n = input("Would you like the output to be in CAPS-LOCK?[y/n]")
        if caps_lock_enable_y_n == 'y':
            globals.caps_lock_enable = True
            break
        elif caps_lock_enable_y_n == 'n':
            globals.caps_lock_enable = False
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    # response delay switch
    while True:
        response_delay_enable_y_n = input("Would you like the system to wait before answering?[y/n]")
        if response_delay_enable_y_n == 'y':
            globals.response_delay_enable = True
            break
        elif response_delay_enable_y_n == 'n':
            globals.response_delay_enable = False
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
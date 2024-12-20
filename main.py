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
from state_transition import *
import globals

def main():
    
    current_state = GreetingState()

    # Before we do anything else, give user option to configure extra stuff (1C)
    extra_config()

    while True:
        # wait 3 seconds before responding if it's enabled
        if globals.response_delay_enable:
            print("Typing...")
            time.sleep(3)

        # let TTS do its thing if it's enabled
        if globals.tts_enable:
            globals.ttsengine.say(current_state.utterance)
            globals.ttsengine.runAndWait()

        # output in all caps if it's enabled
        if globals.caps_lock_enable:
            user_input = input(f"{current_state.utterance.upper()}\n> ")
        else:
            user_input = input(f"{current_state.utterance}\n> ")

        current_state = current_state.handle_input(user_input)

        if isinstance(current_state, GoodbyeState):
            if globals.tts_enable:
                globals.ttsengine.say(current_state.utterance)
                globals.ttsengine.runAndWait()

            print(current_state.utterance)
            time.sleep(1)
            break

# run the dialog system
main()
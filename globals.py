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

tts_enable = False
caps_lock_enable = False
threshold = 3
response_delay_enable = False
ttsengine = pyttsx3.init() # Add 'driverName=nsss' for MacOS compatibility
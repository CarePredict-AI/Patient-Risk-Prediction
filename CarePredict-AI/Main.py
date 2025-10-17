

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import csv
from datetime import datetime


def chatbot():

    print("👋 Welcome to CarePredictAI!")
    name = input("Please enter your name: ")
    print(f"I'm sorry you're feeling unwell, {name}. Let's check what might be going on.")

    age = int(input("Enter your age: "))
    gender = input("Enter your gender (Male/Female): ")
    history = input("Briefly describe your medical history: ")
    lifestyle = input("Describe your lifestyle (e.g., active, sedentary): ")
    symptoms_text = input("Describe your symptoms: ")
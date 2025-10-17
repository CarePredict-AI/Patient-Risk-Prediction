

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import csv
from datetime import datetime

# ============================================================================
# LOAD AND PREPARE DATASET
# ============================================================================
df = pd.read_csv('Full_Patient_Risk_Prediction_Dataset.csv')

# Keep relevant columns
df = df[['Age', 'Gender', 'Symptoms', 'Medical_History', 'Lifestyle', 'Risk_Level']]
df = df.dropna()

# Encode categorical features
le_gender = LabelEncoder()
le_history = LabelEncoder()
le_lifestyle = LabelEncoder()
le_risk = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Medical_History'] = le_history.fit_transform(df['Medical_History'])
df['Lifestyle'] = le_lifestyle.fit_transform(df['Lifestyle'])
df['Risk_Level'] = le_risk.fit_transform(df['Risk_Level'])

# TF-IDF for Symptoms
vectorizer = TfidfVectorizer()
X_symptoms = vectorizer.fit_transform(df['Symptoms'])

# Combine all features into one DataFrame
X_symptoms_df = pd.DataFrame(X_symptoms.toarray(), columns=vectorizer.get_feature_names_out())
X_symptoms_df['Age'] = df['Age'].values
X_symptoms_df['Gender'] = df['Gender'].values
X_symptoms_df['Medical_History'] = df['Medical_History'].values
X_symptoms_df['Lifestyle'] = df['Lifestyle'].values
X = X_symptoms_df
X.columns = X.columns.astype(str)
# Target
y_risk = df['Risk_Level']



# ============================================================================
# TRAIN MODEL
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
model_risk = XGBClassifier(eval_metric='mlogloss')
model_risk.fit(X_train, y_train)




def chatbot():

    print("👋 Welcome to CarePredictAI!")
    name = input("Please enter your name: ")
    print(f"I'm sorry you're feeling unwell, {name}. Let's check what might be going on.")

    age = int(input("Enter your age: "))
    gender = input("Enter your gender (Male/Female): ")
    history = input("Briefly describe your medical history: ")
    lifestyle = input("Describe your lifestyle (e.g., active, sedentary): ")
    symptoms_text = input("Describe your symptoms: ")
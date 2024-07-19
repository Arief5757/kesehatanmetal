import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import joblib

# Load the dataset
path = "C:/Users/LENOVO/Downloads/archive (2)/Mental Health Dataset.csv"
df = pd.read_csv(path)

# Define the function to determine mental health issues
def determine_mental_health_issue(row):
    if row['Mental_Health_History'] == 'Yes' and row['Mood_Swings'] == 'High':
        return 'Depression'
    elif row['Coping_Struggles'] == 'Yes' and row['Changes_Habits'] == 'Yes':
        return 'Anxiety'
    elif row['Social_Weakness'] == 'Yes' and row['Work_Interest'] == 'No':
        return 'Social Anxiety'
    elif row['family_history'] == 'Yes' and row['Days_Indoors'] in ['More than 2 months', '15-30 days', '31-60 days']:
        return 'Isolation'
    elif row['Occupation'] == 'Corporate' and row['Growing_Stress'] == 'High' and row['Days_Indoors'] in ['More than 2 months', '15-30 days', '31-60 days']:
        return 'Corporate Burnout'
    elif row['Occupation'] == 'Student' and row['Growing_Stress'] == 'High' and row['Days_Indoors'] in ['More than 2 months', '15-30 days', '31-60 days']:
        return 'Student Burnout'
    elif row['Occupation'] == 'Others' and row['Mental_Health_History'] == 'Yes' and row['Coping_Struggles'] == 'Yes':
        if row['Gender'] == 'Female':
            return 'Others Job Female Anxiety'
        elif row['Gender'] == 'Male':
            return 'Others Job Male Anxiety'
        else:
            return 'Others Job Anxiety'
    elif row['self_employed'] == 'Yes' and row['Growing_Stress'] == 'High' and row['Coping_Struggles'] == 'Yes':
        return 'Entrepreneurial Stress'
    elif row['self_employed'] == 'Yes' and row['Growing_Stress'] == 'High' and row['Work_Interest'] == 'No':
        return 'Entrepreneurial Burnout'
    elif row['Gender'] == 'Female' and row['Mental_Health_History'] == 'Yes' and row['Coping_Struggles'] == 'Yes':
        return 'Female Anxiety'
    elif row['Gender'] == 'Male' and row['Mental_Health_History'] == 'Yes' and row['Work_Interest'] == 'No':
        return 'Male Social Anxiety'
    else:
        return 'None'

# Add the new column 'mental_health_issue' to the DataFrame
df['mental_health_issue'] = df.apply(determine_mental_health_issue, axis=1)

# Separate features and target
X = df.drop('mental_health_issue', axis=1)  # features
y = df['mental_health_issue']  # target

# Convert categorical features to numeric
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
st.title("Mental Health Issue Prediction with Decision Tree")
st.subheader('Classification Report')
st.text(classification_report(y_test, y_pred))

st.subheader('Confusion Matrix')
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# Save the model
model_filename = 'mental_health_model.pkl'
joblib.dump(model, model_filename)

# Streamlit app for user input and prediction
st.sidebar.header("Input Data")
def user_input_features():
    mental_health_history = st.sidebar.selectbox('Mental Health History', ['Yes', 'No'])
    mood_swings = st.sidebar.selectbox('Mood Swings', ['High', 'Low', 'Moderate'])
    coping_struggles = st.sidebar.selectbox('Coping Struggles', ['Yes', 'No'])
    changes_habits = st.sidebar.selectbox('Changes in Habits', ['Yes', 'No'])
    social_weakness = st.sidebar.selectbox('Social Weakness', ['Yes', 'No'])
    work_interest = st.sidebar.selectbox('Work Interest', ['Yes', 'No'])
    family_history = st.sidebar.selectbox('Family History of Mental Illness', ['Yes', 'No'])
    days_indoors = st.sidebar.selectbox('Days Spent Indoors', ['More than 2 months', '15-30 days', '31-60 days', 'Less than 15 days'])
    occupation = st.sidebar.selectbox('Occupation', ['Corporate', 'Student', 'Others'])
    growing_stress = st.sidebar.selectbox('Growing Stress', ['High', 'Low', 'Moderate'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
    self_employed = st.sidebar.selectbox('Self Employed', ['Yes', 'No'])

    data = {
        'Mental_Health_History': mental_health_history,
        'Mood_Swings': mood_swings,
        'Coping_Struggles': coping_struggles,
        'Changes_Habits': changes_habits,
        'Social_Weakness': social_weakness,
        'Work_Interest': work_interest,
        'family_history': family_history,
        'Days_Indoors': days_indoors,
        'Occupation': occupation,
        'Growing_Stress': growing_stress,
        'Gender': gender,
        'self_employed': self_employed
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Convert user input features to numeric
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Predict mental health issue
prediction = model.predict(input_df)
st.subheader('User Input Data')
st.write(input_df)

st.subheader('Prediction')
st.write(f"Mental Health Issue: {prediction[0]}")

# Show the user the prediction and related information
st.subheader('Prediction Probability')
prediction_proba = model.predict_proba(input_df)
prediction_proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
st.write(prediction_proba_df)

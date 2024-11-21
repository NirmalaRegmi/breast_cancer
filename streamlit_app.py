import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load Data
X = pd.read_csv('selected_features.csv')
y = pd.read_csv('labels.csv').values.ravel()

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit Interface
st.title("Breast Cancer Data Analysis")

# Model Training
if st.button("Train Model"):
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display Results
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

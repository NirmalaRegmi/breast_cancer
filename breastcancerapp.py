import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
@st.cache_data
def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    return df, data.target_names

# Feature Selection
@st.cache_data
def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_features)

# Train Model
def train_model(X_train, y_train, params):
    model = MLPClassifier(**params, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Main Streamlit App
st.title("Breast Cancer Data Analysis")
st.sidebar.header("App Configuration")

# Load Dataset
df, target_names = load_data()
st.write("### Dataset Preview")
st.dataframe(df.head())

# Sidebar Options
st.sidebar.subheader("Data Exploration")
if st.sidebar.checkbox("Show Data Summary"):
    st.write("### Dataset Summary")
    st.write(df.describe())

if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Split Dataset
X = df.drop(columns=["target"])
y = df["target"]
st.write("### Target Distribution")
st.bar_chart(y.value_counts())

# Feature Selection
st.sidebar.subheader("Feature Selection")
k = st.sidebar.slider("Select number of features:", min_value=5, max_value=X.shape[1], value=10, step=1)
X_selected = select_features(X, y, k=k)
st.write(f"### Top {k} Features")
st.dataframe(X_selected.head())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Model Training
st.sidebar.subheader("Model Configuration")
hidden_layer_sizes = st.sidebar.selectbox("Hidden Layer Sizes", [(50,), (100,), (50, 50)])
activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh"])
solver = st.sidebar.selectbox("Solver", ["adam", "sgd"])
alpha = st.sidebar.slider("Regularization Parameter (alpha)", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)

if st.sidebar.button("Train Model"):
    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "alpha": alpha,
    }
    st.write("### Training the Model...")
    model = train_model(X_train, y_train, params)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=target_names))

Breast Cancer Data Analysis 

Project Overview
Welcome to the Breast Cancer Data Analysis project. This project leverages machine learning to classify breast cancer tumors as either malignant or benign. It includes tasks such as data preprocessing, feature selection, and model training. The project features an interactive Streamlit app to visualize the data and understand the model's performance.

Dataset
This project uses a well-known breast cancer dataset containing tumor characteristics such as size, texture, and shape. The goal is to predict whether a tumor is malignant (cancerous) or benign (non-cancerous).

Features (X)
The dataset includes several features that assist in predicting tumor types:

mean_radius: Mean radius of the tumor
mean_texture: Mean texture of the tumor
mean_perimeter: Mean perimeter of the tumor
mean_area: Mean area of the tumor
mean_smoothness: Mean smoothness of the tumor
These features are stored in the features.csv file.

Labels (y)
The labels indicate the tumor's classification:

0: Benign
1: Malignant
These labels are stored in the labels.csv file.

Project Setup
Step 1: Clone the Repository
Clone the project repository to your local machine:

Open your terminal.
Run:
git clone https://github.com/NirmalaRegmi/breast_cancer

Step 2: Set Up a Virtual Environment
To isolate dependencies, set up a virtual environment:

Create the virtual environment:
python -m venv .venv

Step 3: Install Dependencies
Install all necessary libraries by running:
pip install -r requirements.txt

The project uses the following Python libraries:

pandas for data manipulation
scikit-learn for machine learning and feature selection
streamlit to create the interactive app
matplotlib and seaborn for data visualizations
Step 4: Run the Streamlit App
Launch the Streamlit app by running:
streamlit run streamlit_app.py

Once the app starts, open your browser and go to http://localhost:8501/ to interact with it.

Code Walkthrough
Loading and Preprocessing Data
The dataset is loaded using pandas. Features (X) and labels (y) are prepared for machine learning tasks.

X contains tumor characteristics
y contains target labels (malignant or benign)
Feature Selection
The SelectKBest method from scikit-learn is used to identify the most important attributes. This reduces the dataset's dimensionality, improving model accuracy.

Model Training and Tuning
A Multi-Layer Perceptron (MLP) model is trained to classify tumors. Hyperparameters such as hidden layer size and activation functions are optimized using Grid Search with cross-validation.

Streamlit App
The Streamlit app provides a simple interface for users to:

View the dataset
Analyze model performance with metrics such as accuracy and confusion matrix
Input new data to predict tumor classifications
Visualizations
Class Distribution
A bar chart displays the proportion of benign and malignant tumors to provide an overview of dataset balance.

Feature Correlation Heatmap
A heatmap highlights correlations among features, making it easier to identify key relationships.

Feature Importance
A bar plot visualizes the most significant features selected for classification.

Confusion Matrix
A matrix showcases the model's performance, indicating correct and incorrect predictions.

Project Structure
features.csv: Contains tumor features
labels.csv: Contains target labels
streamlit_app.py: Streamlit app for data visualization and interaction

requirements.txt: List of project dependencies
README.md: Project documentation

Usage
Run the Streamlit app using the command: streamlit run streamlit_app.py.
Open your browser and navigate to http://localhost:8501/ to:

View and explore tumor features and their significance
Analyze model performance metrics like accuracy and confusion matrix
Test predictions using new inputs
License
This project is licensed under the MIT License, allowing for both personal and commercial use.

Acknowledgements
Dataset: A commonly used breast cancer dataset in machine learning classification tasks
Streamlit: A user-friendly framework for creating interactive web apps
Scikit-learn: A Python library simplifying the implementation of machine learning algorithms
Seaborn and Matplotlib: Libraries for creating visualizations

Conclusion
This project demonstrates an end-to-end machine learning pipeline for breast cancer classification. From data preprocessing to model deployment, it offers an interactive way to explore machine learning concepts. The Streamlit app makes it accessible and easy to understand for users.

Feel free to explore the project and make it your own. If you have any questions or feedback, please don’t hesitate to reach out.








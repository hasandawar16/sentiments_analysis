# sentiments_analysis
📱 Sentiment Analysis on Electronic Product Reviews
📌 Project Overview

This project performs Sentiment Analysis on electronic product reviews using Machine Learning techniques.
The system classifies customer reviews into Positive, Neutral, or Negative sentiments based on product ratings and review text.

Multiple machine learning models are trained and compared, and the best-performing model is selected automatically.

🎯 Objectives

Convert numerical ratings into sentiment labels
Clean and preprocess review text
Apply TF-IDF feature extraction
Train and compare multiple ML models
Select the best model based on accuracy
Allow user interaction for predictions and analysis

🛠️ Technologies Used

Programming Language: Python
Libraries:
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
NLTK (optional for text handling)
Environment: VS Code / Jupyter Notebook

📂 Dataset

Electronics Product Reviews Dataset
Loaded without headers
Typical columns include:
User ID
Product ID
Rating (1–5)
Review Text (if available)

🎭 Sentiment Mapping
Rating	Sentiment
1–2	Negative
3	Neutral
4–5	Positive
⚙️ Methodology

1️⃣ Data Loading

Dataset loaded using Pandas
Handles cases where review text may not exist

2️⃣ Exploratory Data Analysis

Sentiment distribution visualized using count plots

3️⃣ Text Preprocessing

Lowercasing text
Removing URLs, punctuation, numbers, and HTML tags
Cleaning text for feature extraction

4️⃣ Feature Extraction

TF-IDF Vectorizer
Converts text into numerical features
Limited features for faster processing

5️⃣ Train-Test Split

80% training data
20% testing data
Stratified split to maintain sentiment balance

🤖 Models Implemented

The following models are trained and compared:
Logistic Regression
Multinomial Naive Bayes
Decision Tree Classifier
Random Forest Classifier
The model with the highest accuracy is automatically selected as the best model.

📊 Evaluation Metrics

Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-score)

🧭 User Menu Features

The program provides an interactive menu:
Show Accuracy of the best model
Display Confusion Matrix Heatmap
Predict Sentiment for a New Review
Show Important Words influencing predictions
Exit Program

🔮 Sample Prediction

User enters a product review
System outputs:
Predicted sentiment
Confidence score (if supported by the model)

🚀 Future Enhancements

Use deep learning models (LSTM, BERT)
Add more advanced NLP preprocessing
Deploy as a web application using Flask or Streamlit
Improve performance with larger datasets

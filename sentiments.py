import pandas as pd
import numpy as np
import re
import nltk
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset WITHOUT headers - pandas will use 0, 1, 2 as column names
dataset = pd.read_csv("D:\\document\\ratings_Electronics (1).csv", header=None)

# Check what the data looks like
print("Dataset shape:", dataset.shape)
print("First 10 rows:")
print(dataset.head(10))
print("Column names (auto-assigned):", dataset.columns.tolist())

# Based on common electronic review datasets, the columns are typically:
# Column 0: User ID
# Column 1: Product ID  
# Column 2: Rating (1-5)
# Column 3: Timestamp (optional)
# Column 4: Review Text (if available)

# ADJUST THESE COLUMN NUMBERS BASED ON WHAT YOU SEE ABOVE:
rating_col = 2  # This is usually the rating column (1-5)
text_col = 4    # This is usually the review text column (if available)

# If you don't have review text, we'll need to handle that differently
if dataset.shape[1] <= text_col:
    print("Warning: No review text column found! Only ratings available.")
    print("Creating dummy text from ratings...")
    # Create simple text based on ratings
    dataset['text'] = dataset[rating_col].apply(lambda x: f"rated {x} stars")
    text_col = 'text'  # Use our created text column
    rating_col = rating_col  # Keep the original rating column
else:
    # Rename the columns for clarity
    dataset = dataset.rename(columns={
        rating_col: 'Rating',
        text_col: 'Review_Text'
    })
    rating_col = 'Rating'
    text_col = 'Review_Text'

print(f"Using rating column: {rating_col}")
print(f"Using text column: {text_col}")

# Create sentiment mapping
dataset['Sentiment'] = dataset[rating_col].map({
    1: 'Negative',
    2: 'Negative', 
    3: 'Neutral',
    4: 'Positive',
    5: 'Positive'
})

# Check the distribution
print("\nSentiment distribution:")
print(dataset['Sentiment'].value_counts())

sns.countplot(x=dataset['Sentiment'])
plt.title('Distribution of Sentiments')
plt.show()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'<.*?>+', "", text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), "", text)
    text = re.sub(r'\n', " ", text)
    text = re.sub(r'\w*\d\w*', "", text)
    return text

# Clean the text data
dataset['Cleaned_Text'] = dataset[text_col].apply(clean_text)

# Check if we have enough data after cleaning
print(f"Total samples: {len(dataset)}")

# Use TF-IDF for better results
tfidf = TfidfVectorizer(max_features=2000)  # Reduced for faster processing
X = tfidf.fit_transform(dataset['Cleaned_Text']).toarray()
y = dataset['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train multiple models and find the best one
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial NB": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),  # Limited depth for faster training
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10)  # Reduced for speed
}

best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Track the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Menu system
while True:
    print("\nChoose an option:")
    print("1 - Show Accuracy of Best Model")
    print("2 - Show Heatmap")
    print("3 - Predict Sentiment for a Review")
    print("4 - Show Important Words")
    print("5 - Exit")
    
    choice = input("Enter your choice: ")

    if choice == "1":
        y_pred_best = best_model.predict(X_test)
        print(f"\n{best_model_name} Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

    elif choice == "2":
        y_pred_best = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", 
                   xticklabels=best_model.classes_, 
                   yticklabels=best_model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {best_model_name}")
        plt.show()

    elif choice == "3":
        user_input = input("Enter a product review: ")
        cleaned_input = clean_text(user_input)
        input_vector = tfidf.transform([cleaned_input])
        prediction = best_model.predict(input_vector)
        
        if hasattr(best_model, 'predict_proba'):
            prediction_proba = best_model.predict_proba(input_vector)
            confidence = np.max(prediction_proba)
            print(f"Predicted Sentiment: {prediction[0]}")
            print(f"Confidence: {confidence:.2%}")
        else:
            print(f"Predicted Sentiment: {prediction[0]}")
            print("Confidence score not available for this model")

    elif choice == "4":
        if hasattr(best_model, 'coef_'):
            feature_names = tfidf.get_feature_names_out()
            coefs = best_model.coef_
            
            print("\nTop words for each sentiment:")
            for i, sentiment in enumerate(best_model.classes_):
                top_words_idx = np.argsort(coefs[i])[-10:][::-1]
                top_words = feature_names[top_words_idx]
                top_weights = coefs[i][top_words_idx]
                
                print(f"\n{sentiment}:")
                for word, weight in zip(top_words, top_weights):
                    print(f"  {word}: {weight:.3f}")
                    
        elif hasattr(best_model, 'feature_importances_'):
            feature_names = tfidf.get_feature_names_out()
            importances = best_model.feature_importances_
            top_idx = np.argsort(importances)[-10:][::-1]
            
            print("\nTop important words:")
            for idx in top_idx:
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
        else:
            print("Feature importance not available for this model type")

    elif choice == "5":
        print("Exiting program. Goodbye!")
        break

    else:
        print("Invalid choice, please try again.")
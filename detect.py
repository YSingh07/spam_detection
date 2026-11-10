import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('spam.csv')
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["Message"])

y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)



from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

models = {
    "Naive Bayes": MultinomialNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(),
    "SVM": LinearSVC()
}
y_pred = {
    "Naive Bayes": None,
    "KNN": None,
    "Logistic Regression": None,
    "SVM": None
}

accuracy = {
    "Naive Bayes": None,
    "KNN": None,
    "Logistic Regression": None,
    "SVM": None
}

models["Naive Bayes"].fit(X_train, y_train)
models["KNN"].fit(X_train, y_train)
models["Logistic Regression"].fit(X_train, y_train)
models["SVM"].fit(X_train, y_train)

y_pred["Naive Bayes"] = models["Naive Bayes"].predict(X_test)
y_pred["KNN"] = models["KNN"].predict(X_test)
y_pred["Logistic Regression"] = models["Logistic Regression"].predict(X_test)
y_pred["SVM"] = models["SVM"].predict(X_test)

accuracy["Naive Bayes"] = models["Naive Bayes"].score(X_test, y_test)
accuracy["KNN"] = models["KNN"].score(X_test, y_test)
accuracy["Logistic Regression"] = models["Logistic Regression"].score(X_test, y_test)
accuracy["SVM"] = models["SVM"].score(X_test, y_test)

print(data.head(10))



print("Accuracy of Naive Bayes:", accuracy["Naive Bayes"])
print("Accuracy of KNN:", accuracy["KNN"])  
print("Accuracy of Logistic Regression:", accuracy["Logistic Regression"])
print("Accuracy of SVM:", accuracy["SVM"])

print("Classification Report of Naive Bayes:\n", classification_report(y_test, y_pred["Naive Bayes"]))
print("Classification Report of KNN:\n", classification_report(y_test, y_pred["KNN"]))
print("Classification Report of Logistic Regression:\n", classification_report(y_test, y_pred["Logistic Regression"]))
print("Classification Report of SVM:\n", classification_report(y_test, y_pred["SVM"]))

messages = [
    "Congratulations! You have won 50000. Claim your prize now!",
    "You have been selected for a FREE iPhone. Click now to claim.",
    "URGENT! Your account has won a cash reward of 25000. Call now!",
    "Win exciting prizes by entering our lucky draw. Limited offer!",
    "Get 90 percent OFF on all products. Offer valid today only!",
    "Congratulations! You are the lucky winner of a brand new car!",
    "FREE recharge available! Click the link to claim your reward.",
    "You have won a lottery worth 100000. Contact us immediately.",
    "Exclusive offer! Get a free gift by clicking this link.",
    "Act now! Your chance to win a huge cash prize expires tonight!",
    "Hey, how are you doing today?",
    "Can you send me the notes from yesterday's class?",
    "I'll reach home in about 30 minutes.",
    "Are we meeting at the library today?",
    "Please call me when you are free.",
    "Don't forget to bring your laptop tomorrow.",
    "What time does the class start?",
    "I am going to the market. Do you need anything?",
    "Let's meet for lunch at 1 PM.",
    "Can you help me with my Python project?",
]

messages = vectorizer.transform(messages)

print("Predictions of Naive Bayes:"+ str(models["Naive Bayes"].predict(messages)))

print("Predictions of KNN:"+ str(models["KNN"].predict(messages)))


print("Predictions of Logistic Regression:"+ str(models["Logistic Regression"].predict(messages)))

print("Predictions of SVM:"+ str(models["SVM"].predict(messages)))


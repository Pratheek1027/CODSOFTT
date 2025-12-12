import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("titanic.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Improved Logistic Regression
model = LogisticRegression(
    solver='liblinear',
    max_iter=3000,
    class_weight='balanced',
    C=0.5,               # regularization strength
    penalty='l2'         # default penalty (good for Titanic)
)

model.fit(X_train_scaled, y_train)


# Predict
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

# Create a confusion matrix with custom labels inside each block
group_names = ["Correct: Died", "Wrong: Predicted Survived",
               "Wrong: Predicted Died", "Correct: Survived"]

group_counts = [f"{value}" for value in cm.flatten()]  # numbers
labels = [f"{name}\nCount: {count}" for name, count in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt="", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# Predict for new passenger
new_passenger = [[3, 0, 22, 7.25]]
prediction = model.predict(new_passenger)[0]  # take first value

# Convert numeric result into text
if prediction == 1:
    print("Prediction: This passenger would have SURVIVED 🚢")
else:
    print("Prediction: This passenger would NOT have survived ❌")




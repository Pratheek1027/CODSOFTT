import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------
# 1. LOAD THE DATA
# -------------------------------
df = pd.read_csv("movies.csv", encoding="latin1")   # fixes decoding error
print("Rows, cols:", df.shape)
print(df.head())


# -------------------------------
# 2. CLEANING THE DATA
# -------------------------------

# --- Clean Year column (extract year number)
df["Year"] = df["Year"].str.extract(r"(\d{4})")   # keep only YYYY
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# --- Clean Duration (extract minutes)
df["Duration"] = df["Duration"].str.extract(r"(\d+)")   # keep only number
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

# --- Clean Votes (remove commas and convert)
df["Votes"] = df["Votes"].astype(str).str.replace(",", "")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# --- Fill missing numerical values with median
num_cols = ["Year", "Duration", "Rating", "Votes"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# --- Fill missing category values with mode
cat_cols = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


print("\nMissing values after cleaning:\n", df.isnull().sum())


# -------------------------------
# 3. ENCODING CATEGORICAL COLUMNS
# -------------------------------
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))


# -------------------------------
# 4. SELECT FEATURES AND TARGET
# -------------------------------
X = df[["Year", "Duration", "Genre", "Votes", "Director", "Actor 1", "Actor 2", "Actor 3"]]
y = df["Rating"]


# -------------------------------
# 5. TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------
# 6. RANDOM FOREST MODEL
# -------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 7. PREDICTIONS
# -------------------------------
y_pred = model.predict(X_test)


# -------------------------------
# 8. EVALUATE PERFORMANCE
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMSE:", mse)
print("R² Score:", r2)
print("\nSample Predictions:", y_pred[:10])


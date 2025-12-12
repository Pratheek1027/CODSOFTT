import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("movies.csv", encoding="latin-1")  # Fix Unicode issue

print("Rows, cols:", df.shape)
print(df.head())

# -----------------------------
# 2. Cleaning & Preprocessing
# -----------------------------

# Clean "Year" → remove brackets (2019) → 2019
# -------------------------------
# CLEANING + CONVERTING COLUMNS
# -------------------------------

# Clean Year → extract digits like 2019 from "(2019)"
df["Year"] = df["Year"].astype(str).str.extract(r'(\d+)', expand=False)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Clean Duration → extract 109 from "109 min"
df["Duration"] = df["Duration"].astype(str).str.extract(r'(\d+)', expand=False)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

# Clean Votes → remove commas "12,345" → "12345"
df["Votes"] = df["Votes"].astype(str).str.replace(",", "", regex=True)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# Rating is already numeric, but convert safely
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# --------------------------------
# HANDLE MISSING VALUES (NUMERIC)
# --------------------------------
num_cols = ["Year", "Duration", "Rating", "Votes"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# --------------------------------
# LABEL ENCODE CATEGORICAL COLUMNS
# --------------------------------
from sklearn.preprocessing import LabelEncoder

cat_cols = ["Name", "Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
le = LabelEncoder()

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
    df[col] = le.fit_transform(df[col])

print("\nMissing values after cleaning:\n", df.isnull().sum())

# -----------------------------
# 3. Select Features and Target
# -----------------------------
X = df[["Year", "Duration", "Votes", "Name", "Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]]
y = df["Rating"]

# -----------------------------
# 4. Train-test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Model — Random Forest
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=12
)

model.fit(X_train, y_train)

# -----------------------------
# 6. Predictions & Metrics
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMSE:", mse)
print("R² Score:", r2)

# Show some predictions
print("\nSample Predictions:", y_pred[:10])

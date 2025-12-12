import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ===========================
# 1. Load Dataset
# ===========================
df = pd.read_csv("movies.csv", encoding="latin1")  # handles unicode issue
print("Rows, cols:", df.shape)
print(df.head())

# ===========================
# 2. Clean Numeric Columns
# ===========================

# --- Clean Year → "(2019)" → 2019 ---
df["Year"] = df["Year"].astype(str).str.extract(r'(\d+)', expand=False)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# --- Clean Duration → "109 min" → 109 ---
df["Duration"] = df["Duration"].astype(str).str.extract(r'(\d+)', expand=False)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

# --- Clean Votes → remove commas, convert to number ---
df["Votes"] = df["Votes"].astype(str).str.replace(",", "")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# --- Rating already numeric, but force conversion ---
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")


# ===========================
# 3. Fill Missing Values
# ===========================
num_cols = ["Year", "Duration", "Rating", "Votes"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ["Name", "Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
le = LabelEncoder()

for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
    df[col] = le.fit_transform(df[col])


print("\nMissing values after cleaning:\n", df.isnull().sum())


# ===========================
# 4. Define Features (X) & Target (y)
# ===========================
X = df.drop("Rating", axis=1)   # everything except rating
y = df["Rating"]               # target variable


# ===========================
# 5. Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===========================
# 6. Train Random Forest Model
# ===========================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ===========================
# 7. Model Evaluation (MSE & R²)
# ===========================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMSE:", mse)
print("R² Score:", r2)


# ===========================
# 8. Sample Predictions
# ===========================
print("\nSample Predictions:", y_pred[:10])


# ===========================
# 9. Predict Rating for an INDIVIDUAL Movie
# ===========================
# Example movie:
sample_movie = pd.DataFrame([{
    "Name": le.fit_transform([df["Name"].iloc[0]])[0],   # encode using same pattern
    "Year": 2020,
    "Duration": 120,
    "Genre": le.fit_transform([df["Genre"].iloc[0]])[0],
    "Votes": 50000,
    "Director": le.fit_transform([df["Director"].iloc[0]])[0],
    "Actor 1": le.fit_transform([df["Actor 1"].iloc[0]])[0],
    "Actor 2": le.fit_transform([df["Actor 2"].iloc[0]])[0],
    "Actor 3": le.fit_transform([df["Actor 3"].iloc[0]])[0]
}])

single_pred = model.predict(sample_movie)
print("\nPredicted Rating for New Movie:", single_pred)


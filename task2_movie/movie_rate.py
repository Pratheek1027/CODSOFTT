# movie_rating_rf.py
import pandas as pd
import numpy as np

# modeling and preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# -------- 1. Load data --------
df = pd.read_csv("movies.csv")   # change filename if needed
print("Rows, cols:", df.shape)
print(df.head())

# -------- 2. Quick inspection --------
print("\nMissing values:\n", df.isnull().sum())
print("\nDtypes:\n", df.dtypes)

# -------- 3. Basic cleaning --------
# Example simple fixes: drop rows without Rating or Votes (target and important numeric)
df = df.dropna(subset=["Rating"])   # must have target
# If Votes missing, set to 0
if "Votes" in df.columns:
    df["Votes"] = df["Votes"].fillna(0)

# Duration may be string like '2h 10m' or numeric minutes; try convert to numeric minutes if needed
def parse_duration(x):
    if pd.isna(x):
        return np.nan
    # try numeric first
    try:
        return float(x)
    except:
        s = str(x)
        # handle patterns like "2h 10m", "130 min"
        if "h" in s:
            parts = s.split("h")
            hours = float(parts[0].strip())
            mins = 0
            if "m" in parts[1]:
                mins = float(parts[1].replace("m","").strip())
            return hours*60 + mins
        if "min" in s:
            return float(s.replace("min","").strip())
        # fallback
        return np.nan

if "Duration" in df.columns:
    df["Duration_min"] = df["Duration"].apply(parse_duration)
else:
    df["Duration_min"] = np.nan

# Year: ensure numeric
if "Year" in df.columns:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Create a basic feature: Age of movie
CURRENT_YEAR = pd.Timestamp.now().year
if "Year" in df.columns:
    df["Movie_Age"] = CURRENT_YEAR - df["Year"]
else:
    df["Movie_Age"] = np.nan

# Replace missing numeric fields with median
num_cols = ["Duration_min", "Votes", "Movie_Age"]
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# -------- 4. Feature engineering: Genre, Director, Actors --------
# Genre may be like "Action|Comedy" or comma separated. We'll extract top genres.
if "Genre" in df.columns:
    # normalize separators
    genres_series = df["Genre"].astype(str).str.replace("|", ",", regex=False)
    # explode to count top genres
    all_genres = genres_series.str.split(",").explode().str.strip().value_counts()
    top_genres = all_genres.head(8).index.tolist()   # keep top 8 genres
    print("Top genres:", top_genres)
    for g in top_genres:
        df[f"genre_{g}"] = genres_series.str.contains(g, na=False).astype(int)
else:
    # no genre column
    pass

# Director: keep top 20 directors as binary features, rest = Other
def top_k_one_hot(col, k=20, prefix="dir"):
    if col not in df.columns:
        return []
    topk = df[col].value_counts().head(k).index.tolist()
    for v in topk:
        safe = str(v).replace(" ", "_").replace("/", "_")
        df[f"{prefix}_{safe}"] = (df[col] == v).astype(int)
    return [f"{prefix}_{str(v).replace(' ', '_').replace('/', '_')}" for v in topk]

dir_features = top_k_one_hot("Director", k=20, prefix="director")

# Actors: for Actor 1, Actor 2, Actor 3 keep top 30 across all actor columns
actor_cols = [c for c in df.columns if c.lower().startswith("actor")]
if actor_cols:
    all_actors = pd.Series(dtype=object)
    for c in actor_cols:
        all_actors = all_actors.append(df[c].dropna().astype(str))
    top_actors = all_actors.value_counts().head(30).index.tolist()
    print("Top actors:", top_actors)
    for a in top_actors:
        safe = a.replace(" ", "_").replace("/", "_")
        df[f"actor_{safe}"] = df[actor_cols].apply(lambda row: int(a in row.values.astype(str)), axis=1)
    actor_features = [f"actor_{a.replace(' ', '_').replace('/', '_')}" for a in top_actors]
else:
    actor_features = []

# -------- 5. Select features and target --------
features = []
# numeric features
for c in ["Duration_min", "Votes", "Movie_Age", "Year", "Fare" ]:  # 'Fare' not relevant here; remove if absent
    if c in df.columns:
        features.append(c)
# add genre flags
genre_flags = [c for c in df.columns if c.startswith("genre_")]
features += genre_flags
# add director/actor flags
features += dir_features + actor_features

# remove duplicates and ensure they exist
features = [f for f in features if f in df.columns]
print("Using features:", features)

X = df[features].copy()
y = pd.to_numeric(df["Rating"], errors="coerce")
# drop rows where target is NaN (already dropped earlier but just to be safe)
mask = ~y.isna()
X = X[mask]
y = y[mask]

# -------- 6. Train/test split --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

# optionally scale numeric features for algorithms that need it (RF doesn't strictly need)
num_for_scale = [c for c in ["Duration_min", "Votes", "Movie_Age", "Year"] if c in X_train.columns]
scaler = StandardScaler()
if num_for_scale:
    X_train[num_for_scale] = scaler.fit_transform(X_train[num_for_scale])
    X_test[num_for_scale] = scaler.transform(X_test[num_for_scale])

# -------- 7. Train Random Forest --------
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# -------- 8. Evaluate --------
y_pred = rf.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

# -------- 9. Feature importances --------
fi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Top 15 feature importances:\n", fi.head(15))

plt.figure(figsize=(8,6))
fi.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 feature importances")
plt.show()

# -------- 10. Predict for a single new movie --------
# create a sample dictionary - adapt values to your input
sample = {
    "Duration_min": 120,
    "Votes": 15000,
    "Movie_Age": 5,
    # if year exists
    "Year": CURRENT_YEAR - 5
}
# set genre flags and director/actor flags to 0, then set desired ones
for c in X_train.columns:
    if c not in sample:
        sample[c] = 0

# if movie is Action and Comedy and actor present example:
if "genre_Action" in sample:
    sample["genre_Action"] = 1

sample_df = pd.DataFrame([sample])[X_train.columns]  # order columns
# scale numeric fields if used
if num_for_scale:
    sample_df[num_for_scale] = scaler.transform(sample_df[num_for_scale])

pred = rf.predict(sample_df)[0]
print("Predicted Rating (random forest):", round(pred, 2))

# -------- 11. Optional: Randomized search for hyperparameters (takes time) --------
# Uncomment to run tuning
"""
param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [None, 10, 20, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 0.3, 0.5]
}
rs = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist,
                        n_iter=20, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=1)
rs.fit(X_train, y_train)
print("Best params:", rs.best_params_)
best_rf = rs.best_estimator_
# evaluate best_rf on test set...
"""

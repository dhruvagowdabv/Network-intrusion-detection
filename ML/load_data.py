import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "KDDTrain+.txt"

df = pd.read_csv(data_path, header=None)

columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level"
]

df.columns = columns

# drop difficulty level
df = df.drop(columns=["difficulty_level"])

print(df.head())
print("Columns:", df.columns.tolist())
print("Shape after drop:", df.shape)




# check unique labels
print("Unique labels before:", df["label"].unique())

# convert label to binary
df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)


print("Unique labels after:", df["label"].unique())
print(df["label"].value_counts())


# separate features and label
X = df.drop(columns=["label"])
y = df["label"]

print("Feature shape:", X.shape)
print("Label shape:", y.shape)



categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

print("Categorical columns:", categorical_cols)
print("Number of categorical columns:", len(categorical_cols))

print("Numerical columns count:", len(numerical_cols))




from sklearn.preprocessing import OneHotEncoder

# one-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

encoded_cat = encoder.fit_transform(X[categorical_cols])

encoded_cat_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(categorical_cols)
)


encoded_cat_df.reset_index(drop=True, inplace=True)
numerical_df = X[numerical_cols].reset_index(drop=True)

# combine numerical + encoded categorical
X_encoded = pd.concat([numerical_df, encoded_cat_df], axis=1)

print("Final feature shape after encoding:", X_encoded.shape)



from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


from sklearn.preprocessing import StandardScaler

# feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled train shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# train logistic regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



import joblib


model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "logistic_model.pkl")
joblib.dump(scaler, model_dir / "scaler.pkl")
joblib.dump(encoder, model_dir / "encoder.pkl")

print("Model, scaler, and encoder saved successfully.")


                                                # Phase 2 
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)  # NOTE: unscaled features

# predictions
y_pred_rf = rf_model.predict(X_test)

# evaluation
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))




importances = rf_model.feature_importances_

feature_names = X_encoded.columns

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})


feature_importance_df = feature_importance_df.sort_values(
    by="importance",
    ascending=False
)


print("\nTop Important Features:")
print(feature_importance_df.head(15))
                     
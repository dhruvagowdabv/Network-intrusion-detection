import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data():
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
    df = df.drop(columns=["difficulty_level"])

    # binary label conversion
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    X = df.drop(columns=["label"])
    y = df["label"]

    categorical_cols = ["protocol_type", "service", "flag"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_cat = encoder.fit_transform(X[categorical_cols])

    encoded_cat_df = pd.DataFrame(
        encoded_cat,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    X_encoded = pd.concat(
        [X[numerical_cols].reset_index(drop=True),
         encoded_cat_df.reset_index(drop=True)],
        axis=1
    )

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder

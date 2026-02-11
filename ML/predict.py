import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

rf_model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "encoder.pkl")


def predict_single(sample_dict):
    """
    sample_dict: dictionary of raw feature values (same format as training)
    returns: prediction and probability
    """


    input_df = pd.DataFrame([sample_dict])

    categorical_cols = ["protocol_type", "service", "flag"]
    numerical_cols = [col for col in input_df.columns if col not in categorical_cols]

    encoded_cat = encoder.transform(input_df[categorical_cols])
    encoded_cat_df = pd.DataFrame(
        encoded_cat,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

 
    final_df = pd.concat(
        [input_df[numerical_cols].reset_index(drop=True),
         encoded_cat_df.reset_index(drop=True)],
        axis=1
    )

    # Scale
    final_scaled = scaler.transform(final_df)

    # Predict
    prediction = rf_model.predict(final_scaled)[0]
    probability = rf_model.predict_proba(final_scaled)[0][1]

    return prediction, probability


if __name__ == "__main__":
    
    sample = {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 200,
        "dst_bytes": 300,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 1,
        "srv_count": 1,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 1,
        "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    }

    pred, prob = predict_single(sample)
    print("Prediction:", "Attack" if pred == 1 else "Normal")
    print("Confidence:", prob)

import pandas as pd
from ml.predict import predict_single   # import your model
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
def parse_log_file(file_path):
    df = pd.read_csv(file_path)

    results = []

    for _, row in df.iterrows():
        is_attack_like = (
        row["src_bytes"] > 1000 or
        row["flag"] == "S0"
    )
        feature_dict = {
        "duration": 0,
        "protocol_type": row["protocol"],
        "service": "http",
        "flag": row["flag"],
        "src_bytes": row["src_bytes"],
        "dst_bytes": row["dst_bytes"],

        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
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

        # 🔥 IMPORTANT FIX
        "count": 60 if is_attack_like else 1,
        "srv_count": 10 if is_attack_like else 1,

        "serror_rate": 0.7 if is_attack_like else 0.0,
        "srv_serror_rate": 0.7 if is_attack_like else 0.0,
        "rerror_rate": 0.2 if is_attack_like else 0.0,
        "srv_rerror_rate": 0.2 if is_attack_like else 0.0,

        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,

        "dst_host_count": 60 if is_attack_like else 1,
        "dst_host_srv_count": 10 if is_attack_like else 1,

        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0,

        "dst_host_serror_rate": 0.7 if is_attack_like else 0.0,
        "dst_host_srv_serror_rate": 0.7 if is_attack_like else 0.0,
        "dst_host_rerror_rate": 0.2 if is_attack_like else 0.0,
        "dst_host_srv_rerror_rate": 0.2 if is_attack_like else 0.0
        }
        print("DEBUG FEATURES:", feature_dict)

        result = predict_single(feature_dict)
        print("MODEL OUTPUT:", result)
        results.append({
    "log": row.to_dict(),
    "prediction": result["prediction"],
    "confidence": result["confidence"],
    "alert": result["alert"],
    "reasons": result["reasons"]
})
    return results


if __name__ == "__main__":
    results = parse_log_file("logs/sample_logs.csv")

    for res in results:
        print(res)
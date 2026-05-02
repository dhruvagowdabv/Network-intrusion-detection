import time
import pandas as pd
import requests

from ml.log_parser import parse_log_file


API_URL = "http://127.0.0.1:8000/predict/"


def simulate_realtime(file_path):
    logs = parse_log_file(file_path)

    print("Starting real-time simulation...\n")

    for i, log in enumerate(logs):

        # Extract only feature_dict (not prediction)
        feature_dict = log["log"]  # raw log

        # Convert log → features again
        # (reuse logic if needed later)
        payload = {
            "duration": 0,
            "protocol_type": feature_dict["protocol"],
            "service": "http",
            "flag": feature_dict["flag"],
            "src_bytes": feature_dict["src_bytes"],
            "dst_bytes": feature_dict["dst_bytes"],

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

        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()

            print(f"[{i+1}] Prediction: {result['prediction']} | Confidence: {result['confidence']}")

        except Exception as e:
            print("Error:", e)

        time.sleep(1)  # simulate real-time delay


if __name__ == "__main__":
    simulate_realtime("ml/logs/sample_logs.csv")
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

# check after conversion
print("Unique labels after:", df["label"].unique())
print(df["label"].value_counts())

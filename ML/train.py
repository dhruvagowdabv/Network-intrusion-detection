from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import prepare_data


def train_models():

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder = prepare_data()


                                        # Logistic Regression

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    print("\nLogistic Regression Results")
    print(confusion_matrix(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))


                                    # Random Forest
   
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

   
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)

    print("\nRandom Forest Results")
    print(confusion_matrix(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))


                                                     # Save models

    BASE_DIR = Path(__file__).resolve().parent
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(lr_model, model_dir / "logistic_model.pkl")
    joblib.dump(rf_model, model_dir / "random_forest_model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(encoder, model_dir / "encoder.pkl")

    print("\nModels saved successfully.")


if __name__ == "__main__":
    train_models()

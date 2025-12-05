import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Heart Disease Prediction")

data = pd.read_csv("heartdisease_preprocessing.csv")

# Pisahkan fitur & label
X = data.drop("target", axis=1)
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aktifkan autolog (syarat basic)
mlflow.sklearn.autolog()

# Experiment run MLflow
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # pred = model.predict(X_test)
    # acc = accuracy_score(y_test, pred)

    # # Logging manual hanya untuk metric tambahan
    # mlflow.log_metric("accuracy_manual", acc)
    # mlflow.log_param("n_estimators", 100)
    # mlflow.log_param("random_state", 42)

    # # Simpan artefak model
    # mlflow.sklearn.log_model("model")

    # print("Accuracy:", acc)

    print("Training selesai (basic).")


import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("file:///D:/mlfow/mlflow/mlruns")
mlflow.set_experiment("Iris-Classification")

# 4. Start an MLflow run
with mlflow.start_run():

    # Model parameters
    n_estimators = 100
    max_depth = 5

    # 5. Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 6. Make predictions
    predictions = model.predict(X_test)

    # 7. Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    # 8. Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # 9. Log metric
    mlflow.log_metric("accuracy", accuracy)

    # 10. Log trained model
    mlflow.sklearn.log_model(
        model,
        name="iris_random_forest",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Accuracy:", accuracy)
    print("MLflow run completed!")